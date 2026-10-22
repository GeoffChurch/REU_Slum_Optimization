"""The move onto epistemetry 165f98b, compared statistically with the fourth replay (674dba4)
(temporary: deleted once the comparison has passed and the replays' data with it).

The fourth replay threaded one generator through each run; at 165f98b each draw is keyed by the
order's seed and its place, so its runs cannot be reproduced bit for bit. Both presets were
replayed at the pin on the same designs and orders (screening_check.py `runs`, the acceptance
runs, screen_acceptance.csv), and their runs are compared with the stored ones, run against run on
the same order, by the test pre-registered 2026-10-18 (screening_check.equivalent).

    M=research/roadless/screening_migration.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $M              # the comparison, then GeoffChurch/epistemetry#10's measurement
    uv run python $M self-check   # the comparison shown to see a change, on the stored runs alone

`old_runs` reads epireplay4.parquet's summary columns only (git-ignored, the fourth replay's
rows), never 674dba4's decision records or epireplay.py's code, which does not import at the pin.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import consolidated  # noqa: E402
from screening_check import (  # noqa: E402
    ACCEPTANCE,
    FLIPS,
    ORDERS,
    RESAMPLES,
    comparison,
    designs,
    equivalent,
    in_order,
)

STORED = HERE / "epireplay4.parquet"   # git-ignored: the fourth replay's runs
#: The fourth replay's policies, by its names, as the presets they became.
POLICIES = {"by stake step 1, 5 rates": "realized-loss",
            "adopt/keep/ask step 1, own price": "fewest-wrong"}
SHIFTED = "r2/base"     # self-check: the design whose runs it gives one block more
FLIPPED = 20            # self-check: the runs of FLIPPED_IN whose wrong call it flips 0 -> 1
FLIPPED_IN = "realized-loss"


def old_runs() -> pd.DataFrame:
    """The fourth replay's runs of the comparison set (each preset's `comparison` cells, orders
    0-99), in the acceptance runs' columns that it has (`call` its calls[0], `truth` its full[0]),
    each run's stored order checked to be the fixture's."""
    stored = pd.read_parquet(STORED, columns=["plan", "policy", "veto", "order", "units", "cost",
                                              "calls", "full", "wrong", "asks", "given"])
    presets = consolidated.screen_presets()
    if set(POLICIES.values()) != set(presets):
        raise SystemExit(f"the stored policies stand for {sorted(POLICIES.values())}, the "
                         f"presets are {sorted(presets)}")
    ours = stored["policy"].isin(list(POLICIES)).to_numpy()
    preset = np.array([POLICIES[p] for p in stored["policy"].to_numpy()[ours]])
    plan = stored["plan"].to_numpy()[ours]
    veto = stored["veto"].to_numpy(dtype=bool)[ours]
    cells = {(name, c.plan, c.veto) for name, screen in presets.items()
             for c in comparison(screen)}
    keep = np.array([(p, d, bool(v)) in cells for p, d, v in zip(preset, plan, veto, strict=True)])

    def column(name: str) -> np.ndarray:
        return stored[name].to_numpy()[ours][keep]

    def only(calls: np.ndarray) -> int:
        (c,) = calls.tolist()
        return int(c)

    runs = pd.DataFrame({"preset": preset[keep], "plan": plan[keep], "veto": veto[keep],
                         "order": column("order"), "units": column("units"),
                         "call": [only(c) for c in column("calls")], "cost": column("cost"),
                         "wrong": column("wrong"), "asks": column("asks"),
                         "truth": [only(c) for c in column("full")]})
    if (len(runs) != len(cells) * ORDERS
            or runs.duplicated(["preset", "plan", "veto", "order"]).any()):
        raise SystemExit(f"{STORED.name}: {len(runs)} runs of the comparison set, "
                         f"{len(cells) * ORDERS} expected, each once")
    fixtures = designs()
    moved = [f"{p} {d} order {o}" for p, d, o, given in
             zip(runs["preset"], runs["plan"], runs["order"], column("given"), strict=True)
             if not np.array_equal(np.asarray(given), fixtures[d].orders[o])]
    if moved:
        raise SystemExit(f"{STORED.name}: {len(moved)} runs on another order than the "
                         f"fixture's, first {moved[0]}")
    return in_order(runs)


def wave_undercharge(new: pd.DataFrame) -> None:
    """GeoffChurch/epistemetry#10 on the acceptance runs: per preset, the share of runs a live
    parallel wave would pay more for (`wave_cost` > `cost`), and the mean of (wave_cost - cost)
    over the design's whole cost, over all runs and over the veto runs (C1m4/p64's)."""
    fixtures = designs()
    for p, g in new.groupby("preset", sort=False):
        whole = np.array([float(fixtures[plan].seconds.sum()) for plan in g["plan"]])
        gap = (g["wave_cost"].to_numpy() - g["cost"].to_numpy()) / whole
        over = g["wave_cost"].to_numpy() > g["cost"].to_numpy()
        veto = g["veto"].to_numpy(dtype=bool)
        print(f"#10, {p}: wave_cost > cost on {int(over.sum())} of {len(g)} runs "
              f"({over.mean():.4f}), {int(over[veto].sum())} of the {int(veto.sum())} veto runs "
              f"({', '.join(sorted(set(g.loc[veto, 'plan'])))}); mean (wave_cost - cost) / "
              f"whole cost {gap.mean():.5f} over all runs, {gap[veto].mean():.4f} over the veto "
              f"runs", flush=True)


def main() -> None:
    new = pd.read_csv(ACCEPTANCE)
    failures = equivalent(old_runs(), new, fixtures=designs(), resamples=RESAMPLES, flips=FLIPS)
    print("pass" if not failures else "FAILS on " + "; ".join(failures), flush=True)
    wave_undercharge(new)
    if failures:
        raise SystemExit(1)


class Case(NamedTuple):
    """A self-check case: what it changes, the old and the new runs it compares, and whether
    `equivalent`'s failures on them are the ones expected."""
    what: str
    old: pd.DataFrame
    new: pd.DataFrame
    expected: Callable[[list[str]], bool]


def self_check() -> None:
    """The comparison on the stored runs alone, against themselves (no failure), with SHIFTED's
    runs one block longer in both presets (a shift inside the margins, which the sign-flip test
    must find: Holm rejections and nothing else, SHIFTED's among them), with FLIPPED_IN's wrong
    call flipped 0 -> 1 on its first FLIPPED runs of a design longer than the floor (outside the
    margin: FLIPPED_IN's wrong interval among the failures), with FLIPPED_IN's first run of a
    design no longer than the floor costing 1e-9 more (a draw-free run changed: that run's exact
    failure and nothing else), and with every run of such a design left out of both sides (the
    exact check comparing no run: each preset's refusal and nothing else)."""
    presets = consolidated.screen_presets()
    if FLIPPED_IN not in presets:
        raise SystemExit(f"{FLIPPED_IN} is no preset")
    old = old_runs()
    short = old["plan"].map(lambda p: len(designs()[p].blocks) <= consolidated.SCREEN_FLOOR)
    shifted = old.assign(units=old["units"] + (old["plan"] == SHIFTED))
    pick = old.index[(old["preset"] == FLIPPED_IN) & (old["wrong"] == 0) & ~short][:FLIPPED]
    flipped = old.copy()
    flipped.loc[pick, "wrong"] = 1
    (one,) = old.index[(old["preset"] == FLIPPED_IN) & short][:1]
    costed = old.copy()
    costed.loc[one, "cost"] *= 1 + 1e-9
    at = f"exact: {FLIPPED_IN} {old.loc[one, 'plan']} "
    floorless: pd.DataFrame = old[(~short).to_numpy()].reset_index(drop=True)
    cases = [
        Case(what="the stored runs against themselves", old=old, new=old,
             expected=lambda bad: not bad),
        Case(what=f"units + 1 on every run of {SHIFTED}", old=old, new=shifted,
             expected=lambda bad: (
                 bool(bad) and all(b.startswith("holm: ") for b in bad)
                 and all(any(b.startswith(f"holm: {p} units on {SHIFTED} ") for b in bad)
                         for p in presets))),
        Case(what=f"{FLIPPED_IN}'s wrong flipped on {FLIPPED} runs", old=old, new=flipped,
             expected=lambda bad: any(b.startswith(f"interval: {FLIPPED_IN} wrong ")
                                      for b in bad)),
        Case(what=f"{FLIPPED_IN}'s first run of a design no longer than the floor costing 1e-9 "
                  f"more", old=old, new=costed,
             expected=lambda bad: len(bad) == 1 and bad[0].startswith(at) and "cost" in bad[0]),
        Case(what="no run of a design no longer than the floor", old=floorless, new=floorless,
             expected=lambda bad: (
                 len(bad) == len(presets)
                 and all(any(b.startswith(f"exact: {p} compares no run") for b in bad)
                         for p in presets))),
    ]
    wrong = []
    for case in cases:
        print(f"---- {case.what}", flush=True)
        bad = equivalent(case.old, case.new, fixtures=designs(), resamples=RESAMPLES, flips=FLIPS)
        print(f"{case.what}: " + ("no failure" if not bad else "fails on " + "; ".join(bad)),
              flush=True)
        if not case.expected(bad):
            wrong.append(case.what)
    print("self-check: " + ("as expected" if not wrong else "NOT as expected: " + "; ".join(wrong)),
          flush=True)
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["self-check"]:
        self_check()
    elif not a:
        main()
    else:
        raise SystemExit(__doc__)
