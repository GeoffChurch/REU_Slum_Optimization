"""The arterial's greedy timed step by step, and two such logs compared commit for commit.

    S=research/roadless/arterial_steps.py   # from a tree's root, PYTHONPATH=., single-threaded
    uv run python $S steps <out> <ids,> <max_roads>   # the lineup's arterial, capped at max_roads
    uv run python $S compare <dir_a> <dir_b> <ids,>  # the commits and step times of two logs

`StepLog` patches the engine module's documented instrument seams (`engines._candidate_chords`
and `engines._best_candidate`, which the greedy loop looks up in that namespace) and records one
row per step: the step (the commits before it), the time at its end and its length, the anchors
and candidates it enumerated, whether it committed, the committed road's WKB, and the two highest
gains among the candidates it scored. `steps` proposes the lineup's arterial
(`road_rescore.lineup_arms()[ARTERIAL]`) with `max_roads` replaced, writing
<out>/steps_<id>.parquet after every step, so a killed run keeps its rows.

`common` and `road_rescore` are imported from research/roadless relative to the WORKING
directory, and `reblock` from that tree's environment, so this one file times either tree: the
before/after comparison of the incremental access gain runs it from a base tree and from the
worktree.

`compare`: per block, how many leading steps commit the same road (WKB, exactly), the first step
that does not (with both logs' top-two gains there), and the median and total ratio of step time
(b over a) over the steps both logs share.
"""
from __future__ import annotations

import dataclasses
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from types import TracebackType
from typing import NamedTuple

import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

sys.path.insert(0, "research/roadless")
import common  # noqa: E402
import road_rescore  # noqa: E402

from reblock.methods.arterial import engines  # noqa: E402
from reblock.methods.arterial.reblocker import GreedyArterialReblocker  # noqa: E402


class StepRow(NamedTuple):
    step: int                 # commits before this step
    t_end: float              # s since the log opened
    step_s: float
    anchors: int
    candidates: int
    committed: bool
    commit_wkb: bytes | None
    gain1: float              # the highest gain among the candidates scored (0.0 if none)
    gain2: float              # the second highest


class StepLog:
    """One row per greedy step (the module's docstring), through the engine's seams; `sink`, if
    given, receives the whole log as a frame after every step."""

    def __init__(self, sink: Callable[[pd.DataFrame], None] | None) -> None:
        self.sink = sink
        self.rows: list[StepRow] = []
        self._anchors = self._candidates = 0
        self._t0 = self._last = 0.0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=list(StepRow._fields))

    def __enter__(self) -> StepLog:
        self._chords, self._best = engines._candidate_chords, engines._best_candidate
        chords, best = self._chords, self._best

        def candidate_chords(anchors: list[tuple[float, float]],
                             targets: list[tuple[float, float]]) -> list[LineString]:
            out = chords(anchors, targets)
            self._anchors, self._candidates = len(anchors), len(out)
            return out

        def best_candidate(results: Iterable[tuple[float, BaseGeometry | None]]
                           ) -> tuple[float, BaseGeometry | None]:
            got = list(results)
            gain, real = best(got)
            now = time.perf_counter()
            top = sorted((g for g, r in got if r is not None), reverse=True)[:2] + [0.0, 0.0]
            self.rows.append(StepRow(
                step=sum(r.committed for r in self.rows), t_end=now - self._t0,
                step_s=now - self._last, anchors=self._anchors, candidates=self._candidates,
                committed=real is not None,
                commit_wkb=None if real is None else bytes(real.wkb), gain1=top[0],
                gain2=top[1]))
            self._last = now
            if self.sink is not None:
                self.sink(self.frame())
            return gain, real

        engines._candidate_chords, engines._best_candidate = candidate_chords, best_candidate
        self._t0 = self._last = time.perf_counter()
        return self

    def __exit__(self, kind: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        engines._candidate_chords, engines._best_candidate = self._chords, self._best


def steps(out: Path, ids: list[str], max_roads: int) -> None:
    road_rescore._check_single_threaded()
    out.mkdir(parents=True, exist_ok=True)
    preset = road_rescore.lineup_arms()[road_rescore.ARTERIAL]
    assert isinstance(preset, GreedyArterialReblocker)
    method = dataclasses.replace(preset, max_roads=max_roads)
    print(method, flush=True)
    for b in common.build_blocks(ids):
        path = out / f"steps_{b.block_id}.parquet"
        t = time.perf_counter()
        def sink(df: pd.DataFrame, p: Path = path) -> None:
            road_rescore.write_parquet(df, p)

        with StepLog(sink) as log:
            roads = method.propose(b).roads
        assert roads is not None
        commits = sum(r.committed for r in log.rows)
        print(f"{b.block_id}: {commits} commits, {len(roads)} segments, "
              f"{time.perf_counter() - t:.0f} s", flush=True)


def compare(dir_a: Path, dir_b: Path, ids: list[str]) -> None:
    for bid in ids:
        a = pd.read_parquet(dir_a / f"steps_{bid}.parquet")
        b = pd.read_parquet(dir_b / f"steps_{bid}.parquet")
        n = min(len(a), len(b))
        same = 0
        while same < n and a.commit_wkb.iloc[same] == b.commit_wkb.iloc[same]:
            same += 1
        ratio = b.step_s.iloc[:n].to_numpy() / a.step_s.iloc[:n].to_numpy()
        print(f"{bid}: {same} of {n} shared steps commit the same road "
              f"({int(a.committed.sum())} and {int(b.committed.sum())} commits); step time b/a "
              f"median {float(np.median(ratio)):.3f}, total "
              f"{b.step_s.iloc[:n].sum() / a.step_s.iloc[:n].sum():.3f} "
              f"({a.step_s.iloc[:n].sum():.0f} s -> {b.step_s.iloc[:n].sum():.0f} s)", flush=True)
        if same < n:
            ra, rb = a.iloc[same], b.iloc[same]
            print(f"  first difference at step {same}: a's top gains {ra.gain1!r}, "
                  f"{ra.gain2!r}; b's {rb.gain1!r}, {rb.gain2!r}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 4 and a[0] == "steps":
        steps(Path(a[1]), a[2].split(","), int(a[3]))
    elif len(a) == 4 and a[0] == "compare":
        compare(Path(a[1]), Path(a[2]), a[3].split(","))
    else:
        raise SystemExit(__doc__)
