"""A wave's plan (docs/superpowers/specs/2026-10-21-lease-pool-design.md, 3.4, with the plan's
Ruling 15): each job's estimate, from an estimator Strategy per family of work built once where
the wave is configured, and from it the job's class, lane and node.

- **A family** (`Family`) is a kind of work and the rows that record it: `stored(block, item)` is
  the most demanding stored measure of this work on that block (its largest stored peak and,
  apart from it, its longest stored wall: rows carry no finish time to tell the last), and
  `observed()` every finished job of it on record, for the fit. A peak is a finished job's
  maximum RSS, never a reading while it ran (GeoffChurch/mycooc's lesson 9). The families
  (FAMILIES, the command line's choice by name, resolved once): baselines, presets-gpu and
  presets-cpu (the onecard rows), lenses (a block's runtime baseline: the same mesh and model, the
  spec's section 7), lineup-score, propose, and smoke (Task 10's table).
- **An estimate** (`Estimator.estimate`) is, first, the newest earlier wave's measure of the job
  (`history_of`: their final queues), then the family's stored run, each with the ladder's
  `stored_mem` and `stored_time` margins (1.2 and 1.25 in every preset); with neither, the fit,
  with `fit_mem` (1.5) and no time margin (the worst rate is its margin), over the family's jobs
  and the earlier waves' on the block's side of `fit_min_km2` (FIT_MIN_KM2, desire_fit.out's 1
  km2, where a job's fixed overhead is a small share), the item's own when they are on at least
  `fit_min_blocks` blocks there (FIT_MIN_BLOCKS), else every item's: on a block of at least
  `fit_min_km2`, their worst per-km2 peak (and wall) times the block's area; on a smaller block,
  their largest peak (and wall). Taken over every block, the worst rate is a small block's
  overhead (20 GB/km2 on 0.08 km2 against 8.8 on the large ones), which would class every large
  block past every node.
- **A plan** (`plan_wave`): one job per (block, item); its class the smallest rung holding its
  estimate (pool_ladder.rung_of); the long lane's when its needed minutes pass a fresh lease's
  (pool_queue.fresh_minutes) or its memory the top rung, then needing `fit_mem` times its peak (a
  long run is sized per job, as an OOM past the top rung is: pool_queue); a block's jobs on one
  node (its group: results can differ by machine), and only on a node whose usable memory
  (USABLE_GB: its memory less its tier's reserve) holds every one of them, a pool job's rung or a
  long job's need; the groups to those nodes longest-processing-time first on their pool jobs'
  needed minutes (a long job runs as a run of its own, outside the pools' balance).

Rows are read with pandas, here only: nothing on the cluster imports this module.
"""
from __future__ import annotations

import math
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

import pandas as pd
import pool_queue as pq
from cluster_submit.context import stamped
from pool_ladder import Ladder, class_name, rung_of

from reblock.contracts import Block

#: The fit's floor (Ruling 15, desire_fit.out's choice): the worst per-km2 rate is taken over the
#: jobs on blocks of at least this many km2, where a job's fixed overhead is a small share.
FIT_MIN_KM2 = 1.0

#: An item's own fit needs its jobs on at least this many blocks on the fit's side of FIT_MIN_KM2
#: (Ruling 15 as the controller amended it): on fewer, its worst rate is too few blocks' word, and
#: the family's worst rate stands, the conservative default for the unseen. One fit for every item
#: would class a cheap method's scoring at cycle_desire's 13.8 GB/km2; a misfit of an item's own
#: costs an OOM's promotion, never a drop.
FIT_MIN_BLOCKS = 5


class Observed(NamedTuple):
    """A finished job's measure: its peak RSS (GB) and its wall (minutes), from `source`
    (`rows:<path>`, a stored row's file under the rows root, or `wave:<w>`, an earlier wave's
    final queue)."""

    block: str
    item: str
    peak_gb: float
    minutes: float
    source: str


class Estimate(NamedTuple):
    """A job's estimate: the bases (a peak in GB, a wall in minutes), the margins they are taken
    with, and where the bases came from."""

    peak_gb: float
    minutes: float
    mem_margin: float
    time_margin: float
    source: str

    def gb(self) -> float:
        """The memory the job needs: its peak times its margin."""
        return self.peak_gb * self.mem_margin

    def needed_minutes(self) -> float:
        """The minutes the job needs: its wall times its margin."""
        return self.minutes * self.time_margin


class Family(Protocol):
    """A family of work and the records of it (the module's doc)."""

    @property
    def name(self) -> str: ...

    def stored(self, block: str, item: str) -> Observed | None:
        """The most demanding stored measure of this work on `block` (Ruling 15): its largest
        stored peak and, apart from it, its longest stored wall; None if none is stored."""
        ...

    def observed(self) -> Sequence[Observed]:
        """Every finished job of this work on record, for the fit."""
        ...


# ---- the families --------------------------------------------------------------------------

def _measure(root: Path, path: Path, *, block: object, item: object, peak_gb: float,
             minutes: float) -> Observed:
    """A stored row's measure, checked at this boundary: its block and item strings, its peak and
    wall finite and not negative (else ValueError naming the file); its source the file's path
    under `root`."""
    where = path.relative_to(root).as_posix()
    if not (isinstance(block, str) and isinstance(item, str)):
        raise ValueError(f"{where}: block {block!r} and item {item!r} are not both strings")
    if not all(math.isfinite(v) and v >= 0 for v in (peak_gb, minutes)):
        raise ValueError(f"{where}: a peak of {peak_gb!r} GB in {minutes!r} minutes")
    return Observed(block=block, item=item, peak_gb=peak_gb, minutes=minutes,
                    source=f"rows:{where}")


def _demanding(peak: Observed, wall: Observed) -> Observed:
    """A (block, item)'s stored basis: the largest stored peak (`peak`'s) and, taken apart from
    it, the longest stored wall (`wall`'s: the spec's 3.4, time estimates come the same way; one
    run's wall understated another's by up to 3.5 times); its source names both runs when they
    differ."""
    source = peak.source if wall.source == peak.source else f"{peak.source} (wall {wall.source})"
    return peak._replace(minutes=wall.minutes, source=source)


class _Rows(ABC):
    """A family read from its stored rows, once (`_read`): `stored` a (block, item)'s most
    demanding measure (`_demanding`: its largest peak and its longest wall, the first stored of
    each on a tie); `observed` every run."""

    @abstractmethod
    def _read(self) -> tuple[Observed, ...]: ...

    @cached_property
    def _runs(self) -> tuple[Observed, ...]:
        return self._read()

    @cached_property
    def _most(self) -> dict[tuple[str, str], Observed]:
        peak: dict[tuple[str, str], Observed] = {}
        wall: dict[tuple[str, str], Observed] = {}
        for o in self._runs:
            k = (o.block, o.item)
            if k not in peak or o.peak_gb > peak[k].peak_gb:
                peak[k] = o
            if k not in wall or o.minutes > wall[k].minutes:
                wall[k] = o
        return {k: _demanding(p, wall[k]) for k, p in peak.items()}

    def stored(self, block: str, item: str) -> Observed | None:
        k = (block, item)
        return self._most[k] if k in self._most else None

    def observed(self) -> Sequence[Observed]:
        return self._runs


@dataclass(frozen=True, kw_only=True)
class Baselines(_Rows):
    """A block's baselines (leak, bar, runtime: a job per block and part), from
    `consolidated_rows/baselines/*/<id>.parquet`: item the row's part, its wall the build, mesh
    and model's."""

    root: Path

    @property
    def name(self) -> str:
        return "baselines"

    def _read(self) -> tuple[Observed, ...]:
        out = []
        for f in sorted(self.root.glob("consolidated_rows/baselines/*/*.parquet")):
            t = pd.read_parquet(f, columns=["block", "part", "peak_rss_gb", "t_build", "t_mesh",
                                            "t_model"])
            out += [_measure(self.root, f, block=r["block"], item=r["part"],
                             peak_gb=float(r["peak_rss_gb"]),
                             minutes=(float(r["t_build"]) + float(r["t_mesh"])
                                      + float(r["t_model"])) / 60)
                    for r in t.to_dict("records")]
        return tuple(out)


#: The baseline part whose mesh and model a block's lenses share (the spec's section 7).
RUNTIME = "runtime"

#: The onecard rows' devices: a block's presets on one card, or on one core.
DEVICES = ("gpu", "cpu")


@dataclass(frozen=True, kw_only=True)
class Onecard(_Rows):
    """A block's presets on one `device` (one job per block, item ""), from
    `consolidated_rows/onecard/*/<device>/<id>.parquet`: the presets' whole wall."""

    root: Path
    device: str

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise ValueError(f"no onecard device {self.device!r} (devices: {', '.join(DEVICES)})")

    @property
    def name(self) -> str:
        return f"presets-{self.device}"

    def _read(self) -> tuple[Observed, ...]:
        out = []
        for f in sorted(self.root.glob(f"consolidated_rows/onecard/*/{self.device}/*.parquet")):
            t = pd.read_parquet(f, columns=["block", "device", "peak_rss_gb", "wall"])
            for r in t.to_dict("records"):
                if r["device"] != self.device:
                    raise ValueError(f"{f}: a row of device {r['device']!r} under {self.device}")
                out.append(_measure(self.root, f, block=r["block"], item="",
                                    peak_gb=float(r["peak_rss_gb"]),
                                    minutes=float(r["wall"]) / 60))
        return tuple(out)


@dataclass(frozen=True, kw_only=True)
class Lenses(_Rows):
    """A block's lenses (one job per block, item ""): no peak of their own is stored, so the
    block's runtime baseline (the same mesh and model: the spec's section 7) is their basis, and
    every block's is their fit's; an OOM promotes a job it puts too low."""

    root: Path

    @property
    def name(self) -> str:
        return "lenses"

    def _read(self) -> tuple[Observed, ...]:
        return tuple(o._replace(item="") for o in Baselines(root=self.root).observed()
                     if o.item == RUNTIME)


#: road_rescore.cx_plain's files beside the networks' under an area_* directory (the old metric's):
#: a block's plain mesh, its clearings scored and its homes' weights, not a network's
#: (consolidated.py's lineup readers skip them too).
NOT_NETWORKS = ("__plain.parquet", "__homes.parquet")

#: The lineup roots whose tasks proposed each network and scored it in one process (road_rescore
#: lineup-one: NOTES, "The road methods under free first"), live and superseded alike, so a row's
#: peak is the proposal's as often as the scoring's (62803's cycle_desire: 165 GB there, 16.6 GB
#: scored alone in lineup-f8fa4e53).
PROPOSED_AND_SCORED = frozenset({"lineup-caf14fa2"})


@dataclass(frozen=True, kw_only=True)
class LineupScore(_Rows):
    """A lineup network scored (one job per block and method), from every
    `road_rows/**/area_*/<id>__<item>.parquet` but those under a root of PROPOSED_AND_SCORED: item
    the method, a file's rows (one per budget) one job, its peak and its total wall. An estimate
    measures only the job it estimates (the controller's ruling): a scoring job's basis is a run
    that only scored (lineup-f8fa4e53, lineup-e6131880, the old metric's rows and later roots of
    scoring alone), never one that proposed its network as well, whose peak would class the
    scoring of cycle_desire's large networks past every rung."""

    root: Path

    @property
    def name(self) -> str:
        return "lineup-score"

    def _read(self) -> tuple[Observed, ...]:
        out = []
        for f in sorted(self.root.glob("road_rows/**/area_*/*__*.parquet")):
            if f.name.endswith(NOT_NETWORKS) or PROPOSED_AND_SCORED & set(
                    f.relative_to(self.root).parts):
                continue
            t = pd.read_parquet(f, columns=["block", "method", "peak_rss_gb", "t_total"])
            jobs = set(zip(t["block"], t["method"], strict=True))
            if len(jobs) != 1:
                raise ValueError(f"{f}: rows of {len(jobs)} (block, method)s, not one")
            [(block, method)] = jobs
            out.append(_measure(self.root, f, block=block, item=method,
                                peak_gb=float(t["peak_rss_gb"].max()),
                                minutes=float(t["t_total"].max()) / 60))
        return tuple(out)


#: A proposal's status when it ran to its end (the others failed, timed out or were reused).
OK = "ok"


@dataclass(frozen=True, kw_only=True)
class Proposals(_Rows):
    """A road proposal (one job per block and method), from every
    `road_rows/**/proposals/<id>__<item>.parquet` whose status is ok: item the method, its total
    wall."""

    root: Path

    @property
    def name(self) -> str:
        return "propose"

    def _read(self) -> tuple[Observed, ...]:
        out = []
        for f in sorted(self.root.glob("road_rows/**/proposals/*__*.parquet")):
            t = pd.read_parquet(f)
            ok = t[t["status"] == OK]
            if ok.empty:
                continue                 # a failed run's row may lack a peak or a total wall
            out += [_measure(self.root, f, block=r["block"], item=r["method"],
                             peak_gb=float(r["peak_rss_gb"]),
                             minutes=float(r["total_wall_s"]) / 60)
                    for r in ok[["block", "method", "peak_rss_gb", "total_wall_s"]].to_dict(
                        "records")]
        return tuple(out)


class Basis(NamedTuple):
    peak_gb: float
    minutes: float


#: The smoke waves' jobs (Task 10), by item: the same on every block, taken as stored.
SMOKE_JOBS: dict[str, Basis] = {"quick": Basis(peak_gb=0.5, minutes=1.0),
                                "big": Basis(peak_gb=3.0, minutes=1.0),
                                "oom": Basis(peak_gb=1.0, minutes=1.0),
                                "slow": Basis(peak_gb=0.5, minutes=3.0),
                                "gpu": Basis(peak_gb=2.0, minutes=2.0)}


@dataclass(frozen=True, kw_only=True)
class Smoke:
    """The smoke waves' jobs (SMOKE_JOBS), source `smoke`; an item of no smoke job is refused.
    Nothing to fit (`observed` is empty)."""

    root: Path

    @property
    def name(self) -> str:
        return "smoke"

    def stored(self, block: str, item: str) -> Observed | None:
        if item not in SMOKE_JOBS:
            raise SystemExit(f"smoke has no item {item!r} (items: {', '.join(SMOKE_JOBS)})")
        b = SMOKE_JOBS[item]
        return Observed(block=block, item=item, peak_gb=b.peak_gb, minutes=b.minutes,
                        source="smoke")

    def observed(self) -> Sequence[Observed]:
        return ()


#: The families by name: the command line's choice, resolved once (no default), each built on the
#: rows root (research/roadless of the checkout whose rows it reads).
FAMILIES: dict[str, Callable[[Path], Family]] = {
    "baselines": lambda root: Baselines(root=root),
    "presets-gpu": lambda root: Onecard(root=root, device="gpu"),
    "presets-cpu": lambda root: Onecard(root=root, device="cpu"),
    "lenses": lambda root: Lenses(root=root),
    "lineup-score": lambda root: LineupScore(root=root),
    "propose": lambda root: Proposals(root=root),
    "smoke": lambda root: Smoke(root=root),
}


# ---- estimates -----------------------------------------------------------------------------

def _worst_rate(rates: Sequence[float]) -> float:
    """The fit's rate: the worst of the jobs' per-km2 rates (desire_fit.out's)."""
    return max(rates)


class Pool(NamedTuple):
    """The jobs a fit is taken over, and whose they are: the item's own (`item`) or every item's
    of the family (`family`)."""

    scope: Literal["item", "family"]
    runs: list[Observed]


def _pool(runs: Sequence[Observed], *, item: str, fit_min_blocks: int) -> Pool:
    """Of `runs` (one side of the fit's floor), the fit's: `item`'s own when they are on at least
    `fit_min_blocks` blocks, else the family's every one."""
    mine = [o for o in runs if o.item == item]
    if len({o.block for o in mine}) >= fit_min_blocks:
        return Pool(scope="item", runs=mine)
    return Pool(scope="family", runs=list(runs))


@dataclass(frozen=True, kw_only=True)
class Estimator:
    """A wave's estimates (the module's doc): of `family`'s work, under `ladder`'s margins, from
    `history` (earlier waves' measures of the family, newest wave first: history_of), the family's
    stored runs, or the fit, by `areas` (km2) with its floor `fit_min_km2`, an item's own once its
    jobs are on `fit_min_blocks` blocks."""

    family: Family
    ladder: Ladder
    areas: Mapping[str, float]
    history: Sequence[Observed]
    fit_min_km2: float
    fit_min_blocks: int

    def estimate(self, block: str, item: str) -> Estimate:
        """(block, item)'s estimate: the newest wave's measure of it, else the family's most
        demanding stored run (both with the ladder's stored_mem and stored_time), else the fit
        (`_fitted`)."""
        seen = next((o for o in self.history if (o.block, o.item) == (block, item)), None)
        if seen is None:
            seen = self.family.stored(block, item)
        if seen is None:
            return self._fitted(block, item)
        return Estimate(peak_gb=seen.peak_gb, minutes=seen.minutes,
                        mem_margin=self.ladder.stored_mem, time_margin=self.ladder.stored_time,
                        source=seen.source)

    def _fitted(self, block: str, item: str) -> Estimate:
        """The fit for (`block`, `item`), over the family's jobs and the earlier waves' on the
        block's side of fit_min_km2, `item`'s own when they are on at least fit_min_blocks blocks
        there, else every item's (`_pool`; the source says which): at least fit_min_km2, the
        worst per-km2 peak and wall of those jobs times the block's area; under it, their largest
        peak and wall. Its memory margin is the ladder's fit_mem, its time's none (the worst rate
        is its margin). A block with no area, a job on one, or no job to fit is a SystemExit
        naming it."""
        what = f"{self.family.name}'s fit"
        if block not in self.areas:
            raise SystemExit(f"{block}: nothing of {self.family.name} stored, and no area to fit "
                             "it by (summary_features.csv or the bank)")
        runs = (*self.family.observed(), *self.history)
        unknown = sorted({o.block for o in runs} - set(self.areas))
        if unknown:
            raise SystemExit(f"{what}: no area for the blocks of its jobs "
                             f"{', '.join(unknown[:5])}" + (f" ({len(unknown)} in all)"
                                                            if len(unknown) > 5 else ""))
        area, floor = self.areas[block], self.fit_min_km2
        large = area >= floor
        side = f"{'of at least' if large else 'under'} {floor:g} km2"
        pool = _pool([o for o in runs if (self.areas[o.block] >= floor) == large], item=item,
                     fit_min_blocks=self.fit_min_blocks)
        if not pool.runs:
            raise SystemExit(f"{block} ({area:.2f} km2): {what} has no job on a block {side}")
        if large:
            peak = _worst_rate([o.peak_gb / self.areas[o.block] for o in pool.runs]) * area
            minutes = _worst_rate([o.minutes / self.areas[o.block] for o in pool.runs]) * area
        else:
            peak = max(o.peak_gb for o in pool.runs)
            minutes = max(o.minutes for o in pool.runs)
        blocks = len({o.block for o in pool.runs})
        return Estimate(peak_gb=peak, minutes=minutes, mem_margin=self.ladder.fit_mem,
                        time_margin=1.0, source=f"fit:{pool.scope}:{len(pool.runs)} jobs on "
                                                f"{blocks} blocks {side}")


class _Wave(NamedTuple):
    created: datetime
    wave: str
    observed: list[Observed]


def history_of(waves: Path, family: str) -> list[Observed]:
    """The earlier waves' measures of `family`'s work: of each `waves/<w>/final.db` (a wave's
    final snapshot, the plan's Ruling 14) whose header's family is `family`, each job's last
    attempt that ended DONE (its peak and minutes, source `wave:<w>`), newest wave first by the
    header's `created`. A final.db whose header names another wave than its directory is refused
    (ValueError)."""
    found: list[_Wave] = []
    for path in sorted(waves.glob("*/final.db")):
        h = pq.read_header(path)
        if h.wave != path.parent.name:
            raise ValueError(f"{path} is wave {h.wave}'s")
        if h.family != family:
            continue
        db = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            rows = db.execute(
                "SELECT j.block, j.item, a.peak_gb, a.minutes FROM jobs j JOIN attempts a"
                " ON a.id = (SELECT MAX(id) FROM attempts WHERE job = j.id AND outcome = ?)"
                " ORDER BY j.id", (pq.Outcome.DONE.value,)).fetchall()
        finally:
            db.close()
        found.append(_Wave(created=stamped(h.created), wave=h.wave, observed=[
            Observed(block=r["block"], item=r["item"], peak_gb=r["peak_gb"],
                     minutes=r["minutes"], source=f"wave:{h.wave}") for r in rows]))
    found.sort(key=lambda w: (w.created, w.wave), reverse=True)
    return [o for w in found for o in w.observed]


def areas_of(features_csv: Path, bank: Mapping[str, object]) -> dict[str, float]:
    """Each block's area in km2: summary_features.csv's `area_m2` (consolidated.py's), then, for a
    block the CSV lacks (one consolidated.py has not described: a new region, say), the bank's
    block's boundary (a projected CRS: m2). A block twice in the CSV, an area not positive, or a
    bank's value not a Block is refused."""
    feat = pd.read_csv(features_csv, usecols=["block", "area_m2"])
    if not feat["block"].is_unique:
        raise ValueError(f"{features_csv}: a block twice")
    areas = {str(b): float(a) / 1e6 for b, a in zip(feat["block"], feat["area_m2"], strict=True)}
    for bid, b in bank.items():
        if bid in areas:
            continue
        if not isinstance(b, Block):
            raise TypeError(f"the bank's {bid} is of type {type(b).__name__}, not a Block")
        areas[bid] = float(b.boundary.area) / 1e6
    odd = sorted(b for b, a in areas.items() if not (math.isfinite(a) and a > 0))
    if odd:
        raise ValueError(f"{features_csv}: areas not positive for {', '.join(odd[:5])}")
    return areas


# ---- the plan ------------------------------------------------------------------------------

#: Each node's memory less its tier's reserve (GB): the most one node gives a wave's jobs, a pool
#: worker's rung or a long run's need. A declared table, the spec's 3.4 figures (248 on the GPU
#: nodes and quadro2, 376 less 128; 1,381 on orion): cluster_submit's site and GPU files at the
#: pin (ec79a0eb) hold neither a node's memory (sinfo's) nor its reserve (fitted at run time into
#: the board's tiers.json from other users' jobs). A node not here cannot be planned on.
USABLE_GB: dict[str, float] = {"h100": 248.0, "quadro1": 248.0, "quadro2": 248.0,
                               "tesla1": 248.0, "tesla2": 248.0, "orion": 1381.0}


class Sized(NamedTuple):
    """Where a job's estimate puts it (`_sized`): its lane, its class, the memory it needs (its
    `gb`) and what a node must give it (`hold_gb`: its rung's, a pool worker's; or its own, a long
    run's)."""

    lane: pq.Lane
    cls: str
    gb: float
    hold_gb: float


def _sized(e: Estimate, ladder: Ladder) -> Sized:
    """`e`'s job: a pool's, of the class of the smallest rung holding its gb, when a rung does and
    its needed minutes fit a fresh lease (pool_queue.fresh_minutes); else the long lane's, of that
    rung's class (the top rung's past it), needing `fit_mem` times its peak."""
    rung = rung_of(e.gb(), ladder)
    cls = class_name(ladder.cards, ladder.rungs[-1] if rung is None else rung)
    if rung is not None and e.needed_minutes() <= pq.fresh_minutes(ladder):
        return Sized(lane=pq.Lane.POOL, cls=cls, gb=e.gb(), hold_gb=rung)
    gb = ladder.fit_mem * e.peak_gb
    return Sized(lane=pq.Lane.LONG, cls=cls, gb=gb, hold_gb=gb)


def _able(nodes: Sequence[str], usable_gb: Mapping[str, float], need: float) -> list[str]:
    """The nodes of `nodes` (in order) whose usable memory holds `need` GB."""
    return [n for n in nodes if need <= usable_gb[n]]


def plan_wave(*, wave: str, commit: str, family: str, ladder: Ladder, estimator: Estimator,
              nodes: Sequence[str], usable_gb: Mapping[str, float], ids: Sequence[str],
              items: Sequence[str], line: Callable[[str, str], str], created: str) -> pq.Seed:
    """Wave `wave`'s seed (closed: the `Fixed` feed): a job per (id, item), in that order (`items`
    is `[""]` for a command without `{item}`), job k the k-th from 1, its command `line(block,
    item)` and its estimate `estimator`'s, its lane, class and need `_sized`'s; its group its
    block, every group on one of `nodes` whose usable memory (`usable_gb`, USABLE_GB at the edge)
    holds each of its jobs (a pool job's rung, a long job's need), longest-processing-time first on
    the group's pool jobs' needed minutes (a long job runs as a run of its own: ties by `ids`'
    order, each to the least loaded of the nodes that hold it, ties by `nodes`' order). A group no
    node holds is a SystemExit naming its block and its largest job, never placed where it cannot
    run. The estimator's family and ladder must be `family` and `ladder`, and every node in
    `usable_gb` (else ValueError). A long job estimated past any partition's wall (4287's fitted
    cycle_desire proposal, about 584 h against 7 days) is planned LONG with its estimate as it
    is: refusing or flagging it is the submit's (Task 7)."""
    if family != estimator.family.name or ladder != estimator.ladder:
        raise ValueError(f"wave {wave}: family {family} and ladder {ladder.name}, the estimator's "
                         f"{estimator.family.name} and {estimator.ladder.name}")
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError(f"wave {wave}: nodes {list(nodes)}")
    unknown = [n for n in nodes if n not in usable_gb]
    if unknown:
        raise ValueError(f"wave {wave}: no usable memory known for {', '.join(unknown)} "
                         f"(nodes: {', '.join(usable_gb)})")
    work = [(b, i) for b in ids for i in items]
    est = {w: estimator.estimate(*w) for w in work}
    size = {w: _sized(e, ladder) for w, e in est.items()}
    pooled = dict.fromkeys(ids, 0.0)                     # a group's pool minutes
    largest: dict[str, tuple[str, str]] = {}              # a group's largest job
    for w, s in size.items():
        b = w[0]
        if s.lane == pq.Lane.POOL:
            pooled[b] += est[w].needed_minutes()
        if b not in largest or s.hold_gb > size[largest[b]].hold_gb:
            largest[b] = w
    load = dict.fromkeys(nodes, 0.0)
    node_of: dict[str, str] = {}
    for b in sorted(pooled, key=lambda b: -pooled[b]):   # stable: ties in ids' order
        top = size[largest[b]]
        able = _able(nodes, usable_gb, top.hold_gb)
        if not able:
            raise SystemExit(f"wave {wave}: no node of {', '.join(nodes)} holds {b}'s "
                             f"{largest[b][1] or 'job'} ({top.lane} needing {top.hold_gb:.0f} GB; "
                             f"usable: " + ", ".join(f"{n} {usable_gb[n]:g}" for n in nodes) + ")")
        n = min(able, key=lambda n: load[n])             # the first of the least loaded
        node_of[b] = n
        load[n] += pooled[b]
    jobs = []
    for k, (b, i) in enumerate(work, 1):
        e, s = est[(b, i)], size[(b, i)]
        jobs.append(pq.JobSeed(
            id=k, block=b, item=i, group=b, line=line(b, i), node=node_of[b], cls=s.cls, gb=s.gb,
            minutes=e.needed_minutes(), peak_gb=e.peak_gb, source=e.source, lane=s.lane))
    header = pq.Header(wave=wave, commit=commit, family=family, ladder=ladder, created=created,
                       feed="fixed")
    return pq.Seed(header=header, jobs=tuple(jobs), closed=True)
