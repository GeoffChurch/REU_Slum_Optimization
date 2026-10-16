"""The stored clearings, committed: their format, its writer and its reader.

WHY. The clearing optimizers write their rows into git-ignored directories (relax_rows/,
polish_rows/, clear_rows_*/, pweight_rows/), one parquet a block, and rewrite them in place: the
free-first runs of 2026-10-11 overwrote 12 blocks' rows of two presets. The rows are many GPU-hours.
This module defines their committed core: which buildings each clearing clears, what it achieved
(D, Lens A, the enclosed share, free first's fields, the timings), and how a reader tells that the
block in hand is still the block the positions were made for. clearings_export.py fills it from the
rows; BACKLOG, "The stored clearings made durable".

WHAT. research/roadless/clearings/ holds

    blocks.csv   one line a block: its building count and a fingerprint of its footprints
    runs.csv     one line a run: where it came from and the metric settings its name encodes
    <run>.csv    a run's core, one line a (block, step); <run> is the run's id, the rows' path
                 under research/roadless without its extension (relax_rows/uni/<plan>/D0.05.p2;
                 `.p<p>` the power, `.fw` the Frank-Wolfe rounding of the same rows). A run
                 id ending @<cluster run> holds rows as that cluster run was shipped them
                 (.cluster_submit/runs/<run>/remote/inputs/<block>.tar), where they differ from
                 the local rows since overwritten in place

CSV, ids as space-separated positions. Compressed, as git stores it, the cores weigh what the same
parquet does; a rewritten row shows as one changed line in `git diff`, which is the audit the
in-place overwrites lacked; and nothing is ignored (research/roadless/.gitignore ignores
*.parquet) or needs a library to read. Floats are Python reprs, so they read back exactly.

THE IDS. A clearing is positions into `block.buildings.outlines`, in `Footprints` row order, the
order every optimizer indexes by. They mean something only while the block's data and the reader
that builds it are unchanged. So `blocks.csv` pins each block by its count and a hash of its
outlines' WKB in row order (`fingerprint`, taken through common.build_blocks), and every read hashes
the block it is handed and refuses a mismatch (`StaleClearing`): a block rebuilt with its buildings
reordered, added or moved cannot be handed another block's positions:

    from clearings import Store                        # research/roadless on sys.path
    Store.committed().cleared("relax_rows/uni/<plan>/D0.05.p2", block, step=1)

THE STEPS. A run has one or more rows a block, numbered from 1 by `step`. A single-budget run
(SIMP, the polish) has one. A greedy's rows are nested, each adding buildings to the last (mode
`added`: the clearing through a step is every row's `cleared` up to it, in clearing order); a
floating search's or a swing's rows are each a whole clearing (mode `whole`). `D`, `perm` and the
rest are those of the clearing through the row.

THE COLUMNS (COLUMNS: one fixed set; a file holds the required ones and those with a value in it):

    block         the block's id
    step          the row's number among the block's rows in this run, from 1
    D             the population share cleared (area or building share: the run's `population`),
                  computed at export from the ids and checked to equal the source row's own D
    cleared       positions (see THE STEPS)
    perm          the run's lens at this clearing as the optimizer stored it: 1 - (J_p / J_p0)^(1/p),
                  p the run's `power` (Lens A at D = budget for the power-2 area runs, the P lens
                  at power 1); blank where the source row stored none (collected sets)
    enclosed      B (i): the population share the clearing leaves enclosed (rows made since
                  lifted.Placement, 2026-10-11: free first's; blank before)
    free_D, free_cleared
                  free first's stage: its D and its buildings (held cleared in every search); a
                  block with no home enclosed has free_D 0 and nothing in free_cleared
    t, t_greedy, wall
                  seconds as the source stored them (SIMP: the run; polish: free + greedy +
                  polish, and through the greedy; greedy rows: elapsed at the step; a cluster
                  task's wall)
    h             the mesh spacing the row was scored on, where it differs by row (the oversized
                  blocks' coarse-mesh runs, collected/ov_clearings: a run's own mesh is in runs.csv)
    perm1, P0     greedy-format rows: the P lens at the step and the baseline P of the scorer of
                  the day (a change of scorer moves P0)
    grey          SIMP: the share of buildings still grey in its final design
    greedy_perm, greedy_D, moves, scorings
                  the polish: the greedy clearing it started from, and its own effort
    relaxed_perm, bound_perm, iters, frac_share, frac_cost
                  a Frank-Wolfe run: its relaxation's lens, its bound, iterations, the fractional
                  share of buildings and of cost
"""
from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

HERE = Path(__file__).resolve().parent
COMMITTED = HERE / "clearings"
BLOCKS = "blocks.csv"
RUNS = "runs.csv"


class Exists(FileExistsError):
    """A run's row already stands under its identity (`write_once`)."""


class StaleClearing(ValueError):
    """A committed clearing's block is no longer the block its positions were committed for."""


# ------------------------------------------------------------------ the footprints' fingerprint

class _Outlined(Protocol):
    @property
    def outlines(self) -> Iterable[BaseGeometry]: ...


class BlockLike(Protocol):
    """What a read asks of a block: its id and its buildings' outlines in row order
    (reblock.contracts.Block)."""

    @property
    def block_id(self) -> str: ...

    @property
    def buildings(self) -> _Outlined: ...


def fingerprint(outlines: Iterable[BaseGeometry]) -> str:
    """SHA-256 of the outlines' 2-D little-endian WKB, in row order. WKB is self-delimiting, so
    the concatenation is unambiguous: a moved vertex, two outlines swapped and a building added or
    dropped each change it."""
    geoms = list(outlines)
    if any(g is None for g in geoms):
        raise ValueError("a footprint is missing (None): the block has no row order to pin")
    wkb = shapely.to_wkb(np.asarray(geoms, dtype=object), hex=False, output_dimension=2,
                         byte_order=1)
    return hashlib.sha256(b"".join(wkb)).hexdigest()


@dataclass(frozen=True, kw_only=True)
class BlockPrint:
    """One line of blocks.csv: what a block was when its clearings were committed."""
    block: str
    n: int
    fingerprint: str

    @classmethod
    def of(cls, block: BlockLike) -> BlockPrint:
        outlines = list(block.buildings.outlines)
        return cls(block=block.block_id, n=len(outlines), fingerprint=fingerprint(outlines))


# ------------------------------------------------------------------ the columns


@dataclass(frozen=True, kw_only=True)
class Column:
    """One column of a run file: how its text reads into a typed pandas column and back (a
    cell is a Python or numpy scalar, an int64 array for ids, or blank: None, NaN, <NA>)."""
    name: str
    parse: Callable[[pd.Series[Any]], pd.Series[Any]]
    text: Callable[[Any], str]


def _blank(x: Any) -> bool:
    return x is None or x is pd.NA or (isinstance(x, float) and bool(np.isnan(x)))


def _parse_text(s: pd.Series[Any]) -> pd.Series[Any]:
    return s


def _parse_int(s: pd.Series[Any]) -> pd.Series[Any]:
    return pd.Series([int(x) if x != "" else pd.NA for x in s], index=s.index, dtype="Int64")


def _parse_real(s: pd.Series[Any]) -> pd.Series[Any]:
    return pd.Series([float(x) if x != "" else np.nan for x in s], index=s.index, dtype="float64")


def _parse_ids(s: pd.Series[Any]) -> pd.Series[Any]:
    return pd.Series([np.array(x.split(), dtype=np.int64) for x in s], index=s.index,
                     dtype=object)


def _text_text(x: Any) -> str:
    return str(x)


def _text_int(x: Any) -> str:
    return "" if _blank(x) else str(int(x))


def _text_real(x: Any) -> str:
    return "" if _blank(x) else repr(float(x))


def _text_ids(x: Any) -> str:
    return "" if _blank(x) else " ".join(str(int(i)) for i in x)


TEXT = (_parse_text, _text_text)
WHOLE = (_parse_int, _text_int)
REAL = (_parse_real, _text_real)
IDLIST = (_parse_ids, _text_ids)

COLUMNS: tuple[Column, ...] = tuple(Column(name=n, parse=k[0], text=k[1]) for n, k in (
    ("block", TEXT), ("step", WHOLE), ("D", REAL), ("cleared", IDLIST), ("perm", REAL),
    ("enclosed", REAL), ("free_D", REAL), ("free_cleared", IDLIST), ("t", REAL),
    ("t_greedy", REAL), ("wall", REAL), ("h", REAL), ("perm1", REAL), ("P0", REAL), ("grey", REAL),
    ("greedy_perm", REAL), ("greedy_D", REAL), ("moves", WHOLE), ("scorings", WHOLE),
    ("relaxed_perm", REAL), ("bound_perm", REAL), ("iters", WHOLE), ("frac_share", REAL),
    ("frac_cost", REAL)))
REQUIRED: tuple[str, ...] = ("block", "step", "D", "cleared")
IDS: tuple[str, ...] = tuple(c.name for c in COLUMNS if c.parse is _parse_ids)
_BY_NAME: Mapping[str, Column] = {c.name: c for c in COLUMNS}


def column(name: str) -> Column:
    """A column by the name a file's header gives it: a name outside COLUMNS raises (KeyError),
    never reads as something else."""
    return _BY_NAME[name]


# ------------------------------------------------------------------ the steps (a run's mode)


class Steps(Protocol):
    """How a run's rows make a clearing, resolved once where a run is read (`steps_of`)."""

    def through(self, rows: Sequence[NDArray[np.int64]], step: int) -> NDArray[np.int64]:
        """The positions of the clearing through `step` (from 1), given each row's `cleared` in
        step order."""
        ...


@dataclass(frozen=True)
class Added:
    """Nested rows: each adds buildings to the last."""

    def through(self, rows: Sequence[NDArray[np.int64]], step: int) -> NDArray[np.int64]:
        return np.concatenate(rows[:step])


@dataclass(frozen=True)
class Whole:
    """Each row is a whole clearing."""

    def through(self, rows: Sequence[NDArray[np.int64]], step: int) -> NDArray[np.int64]:
        return rows[step - 1]


def steps_of(mode: str) -> Steps:
    if mode == "added":
        return Added()
    if mode == "whole":
        return Whole()
    raise ValueError(f"unknown mode {mode!r}: added or whole")


# ------------------------------------------------------------------ runs.csv


@dataclass(frozen=True, kw_only=True)
class RunInfo:
    """One line of runs.csv. Blank (None) is "the source does not say"; the settings are the
    ones the source's path (or its rows) names, parsed once at export, never guessed."""
    run: str                  # the id: the file under clearings/ without .csv
    family: str               # simp | simp-fw | polish | greedy | simp-path | grow-prune |
                              # floating | swing | simp-swing | collected
    mode: str                 # added | whole (THE STEPS)
    spec: str                 # the optimizer as the source's path names it (plan, picker, ...)
    power: float | None       # p of J_p
    demand: str               # w (each home injects its population) | pq<q> (PriorityDemand)
    population: str           # area | count: what D and the displacement count
    along: str                # the run's metric, its rows' directory: a raster conductance
                              # (lifted.conductance_of, uni+apwfs ...; a bare along-conductance,
                              # uni | ss... | sl..., the step rule's before 2026-10-13) or the gap
                              # complex's name (gap_clear.complex_of: cx+..., its code's digest)
    search_along: str         # the conductance the search ranked under, when it differed ("")
    mesh: str                 # lifted.MeshSpec.name the run optimized and scored on ("": the
                              # complex's, which has no grid)
    budget: float | None      # the D the run was asked for (a path's rows run to its d_max)
    source: str               # where it came from, under research/roadless
    source_files: int
    source_bytes: int         # on disk, as parquet (SIMP's rows carry its design x, not committed)
    written_first: str        # the source files' mtimes (cluster pulls set them), local time
    written_last: str
    metric: str               # preB | B | mixed: the scorer that made `perm` (rows with `enclosed`
                              # come from lifted.Placement, 2026-10-11 16:21, 7913fe4)
    blocks: int
    rows: int
    ids: int                  # positions in `cleared`
    note: str

    @property
    def steps(self) -> Steps:
        return steps_of(self.mode)


_RUN_FIELDS = tuple(f.name for f in fields(RunInfo))


def _run_text(info: RunInfo) -> list[str]:
    return ["" if v is None else repr(v) if isinstance(v, float) else str(v)
            for v in astuple(info)]


def _real_or_none(s: str) -> float | None:
    return None if s == "" else float(s)


def _run_of(r: Mapping[str, str]) -> RunInfo:
    return RunInfo(
        run=r["run"], family=r["family"], mode=r["mode"], spec=r["spec"],
        power=_real_or_none(r["power"]), demand=r["demand"], population=r["population"],
        along=r["along"], search_along=r["search_along"], mesh=r["mesh"],
        budget=_real_or_none(r["budget"]), source=r["source"],
        source_files=int(r["source_files"]), source_bytes=int(r["source_bytes"]),
        written_first=r["written_first"], written_last=r["written_last"], metric=r["metric"],
        blocks=int(r["blocks"]), rows=int(r["rows"]), ids=int(r["ids"]), note=r["note"])


# ------------------------------------------------------------------ the store


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """Whole or not at all: under a name no glob matches, then renamed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def _read_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _records(frame: pd.DataFrame) -> list[dict[str, str]]:
    """Each row of an all-text frame as {column: text}."""
    return [{str(k): str(v) for k, v in r.items()} for r in frame.to_dict("records")]


@dataclass(frozen=True)
class Store:
    """A committed store of clearings under `root`: `Store.committed()` is research/roadless/
    clearings/. Built once where a program is configured and handed down (a test builds one under
    its tmp_path). Every read goes to the files: nothing is cached to go stale after a write."""
    root: Path

    @classmethod
    def committed(cls) -> Store:
        return cls(COMMITTED)

    def path(self, run: str) -> Path:
        return self.root / f"{run}.csv"

    # -- blocks.csv

    def prints(self) -> dict[str, BlockPrint]:
        return {r["block"]: BlockPrint(block=r["block"], n=int(r["n"]),
                                       fingerprint=r["fingerprint"])
                for r in _records(_read_text(self.root / BLOCKS))}

    def write_prints(self, prints: Iterable[BlockPrint]) -> None:
        rows = sorted(prints, key=lambda p: p.block)
        _write_csv(self.root / BLOCKS, ("block", "n", "fingerprint"),
                   [(p.block, str(p.n), p.fingerprint) for p in rows])

    def check(self, block: BlockLike) -> None:
        """Raises StaleClearing unless `block` is the block its positions were committed for: the
        same number of buildings and the same fingerprint (the outlines, in row order)."""
        try:
            want = self.prints()[block.block_id]
        except KeyError:
            raise KeyError(f"no clearing of block {block.block_id!r} is committed under "
                           f"{self.root}") from None
        outlines = list(block.buildings.outlines)
        if len(outlines) != want.n:
            raise StaleClearing(
                f"block {block.block_id}: its clearings were committed for {want.n} buildings, "
                f"the block in hand has {len(outlines)}: its positions index other buildings")
        got = fingerprint(outlines)
        if got != want.fingerprint:
            raise StaleClearing(
                f"block {block.block_id}: its footprints are no longer the ones its clearings were "
                f"committed for (fingerprint {want.fingerprint[:12]}, now {got[:12]}): the same "
                f"{want.n} buildings moved, changed or reordered, so its positions index other "
                "buildings")

    # -- runs.csv

    def runs(self) -> dict[str, RunInfo]:
        return {i.run: i for i in (_run_of(r) for r in _records(_read_text(self.root / RUNS)))}

    def write_runs(self, infos: Iterable[RunInfo]) -> None:
        rows = sorted(infos, key=lambda i: i.run)
        _write_csv(self.root / RUNS, _RUN_FIELDS, [_run_text(i) for i in rows])

    # -- a run's file

    def table(self, run: str) -> pd.DataFrame:
        """The run's rows, typed (ids as int64 arrays, blank reals NaN, blank ints <NA>), in file
        order. The header must be COLUMNS' names and include REQUIRED."""
        raw = _read_text(self.path(run))
        cols = [column(name) for name in raw.columns]        # an unknown name raises
        missing = [r for r in REQUIRED if r not in raw.columns]
        if missing:
            raise ValueError(f"{self.path(run)}: no column {missing}")
        return pd.DataFrame({c.name: c.parse(raw[c.name]) for c in cols}, index=raw.index)

    def write_table(self, run: str, frame: pd.DataFrame) -> None:
        """Write a run's rows: REQUIRED and every column with a value somewhere, in COLUMNS'
        order, sorted by (block, step). A block's steps must run 1, 2, ... without gaps."""
        unknown = [c for c in frame.columns if c not in _BY_NAME]
        if unknown:
            raise ValueError(f"{run}: columns outside COLUMNS: {unknown}")
        absent = [r for r in REQUIRED if r not in frame.columns]
        if absent:
            raise ValueError(f"{run}: no column {absent}")
        ordered = frame.sort_values(["block", "step"], kind="stable").reset_index(drop=True)
        for block, g in ordered.groupby("block", sort=False):
            if [int(s) for s in g["step"]] != list(range(1, len(g) + 1)):
                raise ValueError(f"{run} {block}: steps {list(g['step'])} are not 1, 2, ...")
        texts = {c.name: [c.text(v) for v in ordered[c.name]]
                 for c in COLUMNS if c.name in ordered.columns}
        keep = [name for name, col in texts.items() if name in REQUIRED or any(col)]
        _write_csv(self.path(run), keep, zip(*(texts[n] for n in keep), strict=True))

    # -- the reader

    def cleared(self, run: str, block: BlockLike, *, step: int) -> NDArray[np.intp]:
        """The positions into `block.buildings.outlines` that `run` cleared on `block`, through
        `step` (from 1; a single-budget run has step 1), in clearing order. Refuses (StaleClearing)
        a block that is not the one the run's positions were committed for; KeyError for a run,
        block or step the store does not hold."""
        self.check(block)
        try:
            info = self.runs()[run]
        except KeyError:
            raise KeyError(f"no run {run!r} in {self.root / RUNS}") from None
        t = self.table(run)
        at = np.flatnonzero(t["block"].to_numpy() == block.block_id)
        if len(at) == 0:
            raise KeyError(f"run {run!r} holds no clearing of block {block.block_id!r}")
        order = np.argsort(t["step"].to_numpy(dtype=np.int64)[at], kind="stable")
        have = [int(s) for s in t["step"].to_numpy(dtype=np.int64)[at][order]]
        if have != list(range(1, len(have) + 1)):
            raise ValueError(f"run {run!r} block {block.block_id!r}: steps {have} are not 1, 2, ...")
        if step not in have:
            raise KeyError(f"run {run!r} block {block.block_id!r}: no step {step}; steps {have}")
        rows = [np.asarray(c, dtype=np.int64) for c in t["cleared"].to_numpy()[at][order]]
        ids = info.steps.through(rows, step)
        n = self.prints()[block.block_id].n
        if len(ids) and (ids.min() < 0 or ids.max() >= n):
            raise ValueError(f"run {run!r} block {block.block_id!r}: positions outside 0..{n - 1}")
        return ids.astype(np.intp)


def cleared_positions(run: str, block: BlockLike, *, step: int) -> NDArray[np.intp]:
    """`Store.committed().cleared(run, block, step=step)`."""
    return Store.committed().cleared(run, block, step=step)


# ------------------------------------------------------------------ the runs' rows, written once

def check_new(path: Path, *, replace: bool) -> None:
    """Refuses (`Exists`) to start a run whose row `path` already stands, unless `replace` (the
    caller naming a deliberate replacement): the optimizers' no-overwrite guard (BACKLOG, "The
    stored clearings made durable"; NOTES, "The consolidated pass"). A row's path is its identity:
    the block, the preset or method, the metric's name, the budget and the seed (each optimizer's
    rows directory spells them: relax.rows_of, polish_greedy.rows_of, clear.rows_dir, ...), so a
    finished identity re-run is an error, before any work."""
    if path.exists() and not replace:
        raise Exists(f"{path}: a row stands under this identity (block, preset or method, metric, "
                     "budget, seed); a replacement is named, never taken (--replace)")


def _exclusive(tmp: Path, path: Path) -> None:
    """`tmp` to `path` if nothing stands there (a hard link: atomic, and it fails if `path`
    exists, so two runs racing for one identity cannot both write)."""
    os.link(tmp, path)


def write_once(df: pd.DataFrame, path: Path, *, replace: bool) -> None:
    """A run's rows to `path` whole (a temp name no pull matches, then into place) and only if no
    row stands there, unless `replace` (check_new): a second write of one identity raises
    `Exists` and leaves the first. The guard: clearings_check.py `overwrite` (fault `no-check`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp)
    try:
        if replace:
            os.replace(tmp, path)
            return
        try:
            _exclusive(tmp, path)
        except FileExistsError:
            raise Exists(f"{path}: a row stands under this identity; not replaced (--replace "
                         "names a replacement)") from None
    finally:
        tmp.unlink(missing_ok=True)


def replace_flag(argv: list[str]) -> tuple[list[str], bool]:
    """A command line less its `--replace` (the caller naming a deliberate replacement of the
    rows its run would write), and whether it was given."""
    return [a for a in argv if a != "--replace"], "--replace" in argv
