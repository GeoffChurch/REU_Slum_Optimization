"""Export the stored clearings' core into research/roadless/clearings/ (format: clearings.py).

    PYTHONPATH=. uv run python research/roadless/clearings_export.py inventory
    PYTHONPATH=. uv run python research/roadless/clearings_export.py export [--replace RUN[,RUN...]]

From the repo root, on the CPU, single-threaded (OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1). `inventory` lists what the sources hold and writes nothing.
`export` reads every source run, builds the blocks they name (common.build_blocks), checks each row
against its block (the building count the row recorded; positions in range and distinct; D
recomputed from the positions equal to the row's own D) and against the run's settings (the path's
against the row's own columns), and writes the committed core.

The sources are the rows of this code's metric (`this_code`: a gap complex metric's name that
gap_clear.complex_of reads back, its code's digest this code's), each a Family (a Strategy resolved
here, where the directories are named) that finds its runs, parses the settings their names encode
and reads one file into core rows:

    SimpFamily       relax_rows/<metric>/<plan>/D<budget>/<block>_p<power>.parquet: SIMP's clearing
                     (simp_cleared; run <dir>.p<power>) and, in the Frank-Wolfe plans' rows, the
                     rounding of the relaxation (cleared; run <dir>.p<power>.fw)
    PolishFamily     polish_rows/<metric>/<picker>.P<tries>w<width>[x2][E]/D<budget>/..._p<power>
    GreedyFamily     clear_rows_<spec>_p<power>_<metric>/<block>.parquet (clear.rows_dir): the
                     searches that write the greedy's format (nested rows `cleared`, or a whole
                     clearing a row `clearing`)
    PweightFamily    pweight_rows/<metric>/simp/<objective>/<block>.parquet (priority-weighted P)
    ShippedFamily    rows of the stores above that a cluster run was shipped in its input tars
                     (.cluster_submit/runs/<run>/remote/inputs/<block>.tar) and that differ from
                     the local row now or are gone from it: the rows a later run overwrote in
                     place, read by the family of their store as run <run id>@<cluster run>

Every other metric's rows are RECORDS, read by no code: the retired raster's (its conductances'
names, uni+apwfs and the rest; clear_rows_<picker>_h<h>...; pweight_rows/simp, fw, score and
uni+apwfs; the clearings the studies gathered into one parquet, RECORD_PARQUETS) and another code's
complex (phase 1's undigested names). Their runs were exported before the raster's retirement
(2026-10-15) and stay committed; `inventory` counts the records' directories.

Every source column is committed, checked, or deliberately dropped with a reason (a Slot); one that
is none of them raises, so a column a new optimizer adds is never silently left out. So does a
directory under research/roadless that looks like a rows store and is neither a store's, nor in
EXCLUDED with its reason, and a top-level parquet holding clearings that is not a known record.
Rows are read without SIMP's design x (a float a building, half the bytes of the rows, not the
clearing). Every row states `enclosed` (B (i)): a run is the B scorer's (RunInfo.metric `B`).

It is re-runnable: a (run, block) the sources no longer hold stays committed, one the sources hold
and the store does not is added, and one that differs raises (the in-place overwrite this store
exists for) unless the run is named in --replace, when the new row replaces it and the old one is
in git's history. A block whose footprints changed since its first export raises too.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import sys
import tarfile
import tempfile
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import gap_clear  # noqa: E402
import gap_factor  # noqa: E402
import metrics  # noqa: E402
from clearings import BlockPrint, RunInfo, Store  # noqa: E402

D_TOL = 1e-9        # a row's D against the one recomputed from its positions (sums in another order)

NUM = r"[0-9]+(?:\.[0-9]+)?"

# Directories that look like rows stores and are not clearings, and the file-level equivalents.
EXCLUDED: Mapping[str, str] = {
    "insulator_rows": "scenario scores of the insulator world; the clearings it scored are "
                      "insulator_rows/walls_clearings.parquet, a record (RECORD_PARQUETS)",
    "road_rows": "road networks and their scores (BACKLOG 4), no clearing",
    "transparency_rows": "road networks and their scores (BACKLOG 5), no clearing",
    "res_rows": "mesh-resolution rescoring of stored clearings: scores only",
    "rows_h0.5_ell3_K8": "the 220-block study's road-network scores, no clearing",
    "rows_h0.5_ell3_K8_area": "the 220-block study's road-network scores, no clearing",
    "rows_h0.5_ell3_K8_area_p1only": "the 220-block study's road-network scores, no clearing",
    "aperture_rows": "the grid fix's study (NOTES, 'The grid fix: steps that conduct by their "
                     "open width'): scores under the step rules it compared, and SIMP's clearing "
                     "under Fws on one block (simp/Fws), a study's record (summary_*.csv)",
    "width_rows": "the effective-width study (NOTES, 'Uniform conductance with an effective "
                  "width'): scores, and SIMP's clearings under B_r and D_s on the sentinel "
                  "(simp/<variant>), a study's record (summary_*.csv)",
    "cross_rows": "the cross-checks of the complex against the raster (phase 2b): the raster's "
                  "scores of the road prefixes and stored clearings, no clearing",
    "gap_rows": "the gap complex's study and the runtime's calibration (NOTES, 'The gap complex' "
                "and 'The complex as the runtime'): scores; simp/ the width forms' SIMP clearings "
                "on the rebuilt complex (a study's record); runtime/preset/ the acceptance's "
                "per-device copies of the complex presets' rows (the same clearings as "
                "relax_rows/<metric> and polish_rows/<metric>, exported) and the dominated "
                "+relaxlumped's (deleted; its numbers in NOTES)",
}
# Top-level parquets holding clearings: every one the retired raster's, a record read by no code
# (their sets were exported to clearings/ as runs collected/<file>/<name> before its retirement).
RECORD_PARQUETS: Mapping[str, str] = {
    "fine_clearings.parquet": "clearings gathered for resolution_check.py (its clearings.parquet "
                              "input; runs fine25, fine35), 2026-10-10",
    "cheap_a5_clearings.parquet": "the cheap preset's clearings on the composite mesh a5x8, "
                                  "gathered for resolution_check.py, 2026-10-08",
    "ov_clearings.parquet": "the oversized blocks' coarse-h runs' clearings, kept when those runs "
                            "were superseded (NOTES, 'Local coarsening: built'), 2026-10-08",
    "ovgoal_clearings.parquet": "the oversized blocks' clearings gathered for resolution_check.py "
                                "on goal meshes, 2026-10-08",
    "res_clearings.parquet": "the five clearings of each of the 59 large blocks gathered for "
                             "resolution_check.py (run res-mesh; NOTES, 'Local coarsening: "
                             "built'), 2026-10-08",
    "three_clearings.parquet": "14401, 32841 and 7851's clearings gathered for resolution_check.py "
                               "(runs three-res, three-res2; NOTES, 'Local coarsening: built'), "
                               "2026-10-08",
    "relax220_p1.parquet": "a copy of relax_rows/uni/fw40.q3.i20.t1e-05/D0.1.p1's rows",
    "relax220_p2.parquet": "a copy of relax_rows/uni/fw40.q3.i20.t1e-05/D0.1.p2's rows",
}
# Source columns of the clearing lists: a top-level parquet holding one is a known record.
ID_COLUMNS = ("cleared", "simp_cleared", "clearing")


def this_code(metric: str) -> bool:
    """Whether a rows directory's metric name is this code's (gap_clear.complex_of reads it back:
    a gap complex metric whose code's digest is this code's, `@e<eps>` its translucent search).
    Any other is a record no code reads: the retired raster's names, another code's complex."""
    if not metric.startswith(metrics.COMPLEX):
        return False
    try:
        gap_clear.complex_of(metric, factorization=gap_factor.SuperLU())
    except ValueError:
        return False
    return True


# ------------------------------------------------------------------ settings and sources

Row = dict[str, Any]       # core columns (clearings.COLUMNS) and `n`, the row's own building count


@dataclass(frozen=True, kw_only=True)
class Settings:
    """What a run's path and rows say of its metric, before any row is read (RunInfo's fields
    but the statistics)."""
    run: str
    family: str
    mode: str
    spec: str
    power: float | None
    demand: str
    population: str
    along: str
    search_along: str
    mesh: str
    budget: float | None
    source: str
    note: str


@dataclass(frozen=True, kw_only=True)
class Source:
    """One run at the source: its files, the blocks they hold and how one file reads into rows."""
    settings: Settings
    files: tuple[Path, ...]
    block_ids: tuple[str, ...]
    read: Callable[[Path], list[Row]]


class Family(Protocol):
    def sources(self, root: Path) -> list[Source]: ...


def block_of(f: Path) -> str:
    return re.sub(r"_p[0-9.]+$", "", f.stem)


def ids(x: Any) -> np.ndarray:
    return np.asarray([] if x is None else x, dtype=np.int64)


@dataclass(frozen=True, kw_only=True)
class Slot:
    """How a source row becomes core rows: its columns committed (source -> core name), only
    checked against the run, deliberately not committed (why), or committed by another run of the
    same file (`sibling`)."""
    commit: Mapping[str, str]
    check: tuple[str, ...]
    drop: Mapping[str, str]
    sibling: tuple[str, ...] = ()

    def known(self) -> set[str]:
        return {*self.commit, *self.check, *self.drop, *self.sibling}


def audit(columns: Iterable[str], slot: Slot, where: Path) -> None:
    extra = sorted(set(columns) - slot.known())
    if extra:
        raise ValueError(f"{where}: columns {extra} are neither committed, checked nor dropped "
                         "(clearings_export.py's Slot): classify them")


def read_frame(f: Path, drop: Iterable[str]) -> pd.DataFrame:
    """The file without the columns never read (SIMP's design x: a float a building)."""
    skip = set(drop)
    return pd.read_parquet(f, columns=[n for n in pq.read_schema(f).names if n not in skip])


def commit(r: pd.Series, slot: Slot) -> Row:
    out: Row = {}
    for src, name in slot.commit.items():
        if src in r.index:
            v = r[src]
            out[name] = ids(v) if name in clearings.IDS else v
    return out


def same(what: str, got: Any, want: Any, where: Path) -> None:
    if got != want:
        raise ValueError(f"{where}: {what} is {got!r}, its run's is {want!r}")


# -- SIMP (relax.py `one`): the incumbent's clearing and the Frank-Wolfe rounding

SIMP = Slot(
    commit={"simp_D": "D", "simp_perm": "perm", "simp_cleared": "cleared", "simp_grey": "grey",
            "t": "t", "free_D": "free_D", "free_cleared": "free_cleared", "enclosed": "enclosed",
            "wall": "wall"},
    check=("block", "n", "power", "plan", "along", "demand", "objective", "item", "budget"),
    drop={"simp_x": "SIMP's final design, a float a building: not the clearing, and half the "
                    "rows' bytes", "x": "the Frank-Wolfe iterate, a float a building",
          "host": "the machine", "peak_rss_gb": "the machine",
          "device": "the solves' device (gap_clear.ComplexMetric.device): scores alike to the "
                    "floating point, not the run's identity"},
    sibling=("iters", "relaxed_perm", "bound_perm", "rounded_perm", "rounded_D", "frac_share",
             "frac_cost", "cleared"))
FW = Slot(
    commit={"rounded_D": "D", "rounded_perm": "perm", "cleared": "cleared", "t": "t",
            "iters": "iters", "relaxed_perm": "relaxed_perm", "bound_perm": "bound_perm",
            "frac_share": "frac_share", "frac_cost": "frac_cost"},
    check=SIMP.check,
    drop=SIMP.drop,
    sibling=("simp_D", "simp_perm", "simp_cleared", "simp_grey", "free_D", "free_cleared",
             "enclosed", "wall"))


def simp_reader(slot: Slot, meta: Settings) -> Callable[[Path], list[Row]]:
    """Reads a relax.py `one` file with `slot`; the row's own columns must say what the run's
    path says (a row of the pweight runs carries its own power, demand and plan)."""
    def read(f: Path) -> list[Row]:
        df = read_frame(f, drop=("simp_x", "x"))
        audit(df.columns, slot, f)
        [r] = (row for _, row in df.iterrows())
        same("block", r["block"], block_of(f), f)
        if "plan" in r.index:
            same("plan", r["plan"], meta.spec, f)
        if "along" in r.index:
            same("along", r["along"], meta.along + (f"@{meta.search_along}"
                                                   if meta.search_along else ""), f)
        if "demand" in r.index:
            same("demand", r["demand"], meta.demand, f)
        if "power" in r.index and meta.power is not None:
            same("power", float(r["power"]), meta.power, f)
        if "budget" in r.index:
            same("budget", float(r["budget"]), meta.budget, f)
        row = commit(r, slot)
        row.update(block=r["block"], step=1, n=int(r["n"]))
        return [row]
    return read


def split_along(token: str) -> tuple[str, str]:
    """(the run's metric, its translucent search's `e<eps>` where it has one): a metric directory's
    name, which this_code accepted."""
    metric, _, search = token.partition("@")
    return metric, search


def _metric_dirs(root: Path, store: str) -> list[Path]:
    """The metric directories under a store (relax_rows/<metric>, ...) whose metric is this
    code's."""
    base = root / store
    return sorted(d for d in base.iterdir() if d.is_dir() and this_code(d.name)) \
        if base.is_dir() else []


def _by_power(d: Path) -> dict[str, list[Path]]:
    by_power: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(d.glob("*.parquet")):
        p = re.fullmatch(r".+_p(?P<power>[0-9.]+)", f.stem)
        if p is None:
            raise ValueError(f"{f}: not <block>_p<power>.parquet")
        by_power[p["power"]].append(f)
    return by_power


@dataclass(frozen=True)
class SimpFamily:
    def sources(self, root: Path) -> list[Source]:
        out: list[Source] = []
        for md in _metric_dirs(root, "relax_rows"):
            along, search = split_along(md.name)
            for d in sorted(md.glob("*/D*")):
                rel = d.relative_to(root).as_posix()
                m = re.fullmatch(rf"relax_rows/[^/]+/(?P<plan>[^/]+)/D(?P<budget>{NUM})", rel)
                if m is None:
                    raise ValueError(f"{rel}: not relax_rows/<metric>/<plan>/D<budget>")
                for power, files in _by_power(d).items():
                    base = Settings(run=f"{rel}.p{power}", family="simp", mode="whole",
                                    spec=m["plan"], power=float(power), demand="w",
                                    population="area", along=along, search_along=search, mesh="",
                                    budget=float(m["budget"]), source=rel, note="")
                    out.append(Source(settings=base, files=tuple(files),
                                      block_ids=tuple(block_of(f) for f in files),
                                      read=simp_reader(SIMP, base)))
                    fw_files = tuple(f for f in files
                                     if "rounded_perm" in pq.read_schema(f).names)
                    if fw_files:
                        fw = replace(base, run=f"{rel}.p{power}.fw", family="simp-fw",
                                     note="the relaxation's rounding (round_by_x), not SIMP's "
                                          "answer")
                        out.append(Source(settings=fw, files=fw_files,
                                          block_ids=tuple(block_of(f) for f in fw_files),
                                          read=simp_reader(FW, fw)))
        return out


# -- the polish (polish_greedy.py)

POLISH = Slot(
    commit={"perm": "perm", "D": "D", "cleared": "cleared", "greedy_perm": "greedy_perm",
            "greedy_D": "greedy_D", "moves": "moves", "scorings": "scorings", "t": "t",
            "t_greedy": "t_greedy", "free_D": "free_D", "free_cleared": "free_cleared",
            "enclosed": "enclosed"},
    check=("block", "n", "picker"),
    drop={"device": SIMP.drop["device"],
          "mesh": "the retired raster's grid (lifted.MeshSpec.name), h0.5 in the complex's rows "
                  "made before phase 3: a placeholder the complex ignored"})


@dataclass(frozen=True)
class PolishFamily:
    def sources(self, root: Path) -> list[Source]:
        out: list[Source] = []
        for md in _metric_dirs(root, "polish_rows"):
            along, search = split_along(md.name)
            for d in sorted(md.glob("*/D*")):
                rel = d.relative_to(root).as_posix()
                m = re.fullmatch(rf"polish_rows/[^/]+/(?P<spec>(?P<picker>S{NUM}cat(?:w[0-9]+)?)"
                                 rf"\.P[0-9]+w[0-9]+(?:x2)?E?)/D(?P<budget>{NUM})", rel)
                if m is None:
                    raise ValueError(f"{rel}: not polish_rows/<metric>/<picker>.P<t>w<w>[x2][E]/"
                                     "D<budget>")
                for power, files in _by_power(d).items():
                    meta = Settings(run=f"{rel}.p{power}", family="polish", mode="whole",
                                    spec=m["spec"], power=float(power), demand="w",
                                    population="area", along=along, search_along=search, mesh="",
                                    budget=float(m["budget"]), source=rel, note="")
                    out.append(Source(settings=meta, files=tuple(files),
                                      block_ids=tuple(block_of(f) for f in files),
                                      read=self._reader(m["picker"])))
        return out

    @staticmethod
    def _reader(picker: str) -> Callable[[Path], list[Row]]:
        def read(f: Path) -> list[Row]:
            df = pd.read_parquet(f)
            audit(df.columns, POLISH, f)
            [r] = (row for _, row in df.iterrows())
            same("block", r["block"], block_of(f), f)
            same("picker", r["picker"], picker, f)
            row = commit(r, POLISH)
            row.update(block=r["block"], step=1, n=int(r["n"]))
            return [row]
        return read


# -- the searches that write the greedy's format (search.py, relax.py `path`; clear.rows_dir)

GREEDY = Slot(
    commit={"step": "step", "D": "D", "perm": "perm", "perm1": "perm1", "cleared": "cleared",
            "clearing": "cleared", "P0": "P0", "t": "t", "enclosed": "enclosed"},
    check=("block", "n"),
    drop={"device": SIMP.drop["device"]})
GREEDY_DIR = re.compile(rf"^clear_rows_(?P<picker>[^_]+)_p(?P<power>{NUM})_(?P<along>cx\+[^/]+)$")


def greedy_family(picker: str) -> str:
    if picker.startswith("SIMP"):
        return "simp-swing" if "+" in picker else "simp-path"
    for prefix, family in (("GP", "grow-prune"), ("FL", "floating"), ("SW", "swing")):
        if picker.startswith(prefix):
            return family
    raise ValueError(f"unknown search {picker!r}")


@dataclass(frozen=True)
class GreedyFamily:
    def sources(self, root: Path) -> list[Source]:
        out: list[Source] = []
        for d in sorted(root.glob("clear_rows_*")):
            m = GREEDY_DIR.fullmatch(d.name)
            if not d.is_dir() or m is None or not this_code(m["along"]):
                continue                    # a record (the raster's greedy rows), or no store
            files = tuple(sorted(d.glob("*.parquet")))
            if not files:
                continue                    # an empty directory (a run that wrote nothing)
            names = pq.read_schema(files[0]).names
            mode = "whole" if "clearing" in names else "added"
            family = greedy_family(m["picker"])
            along, search = split_along(m["along"])
            budget = re.search(rf"D({NUM})$", m["picker"])
            meta = Settings(
                run=d.name, family=family, mode=mode, spec=m["picker"],
                power=float(m["power"]), demand="w", population="area", along=along,
                search_along=search, mesh="",
                budget=float(budget[1]) if budget else None, source=d.name, note="")
            out.append(Source(settings=meta, files=files,
                              block_ids=tuple(block_of(f) for f in files),
                              read=self._reader(meta)))
        return out

    @staticmethod
    def _reader(meta: Settings) -> Callable[[Path], list[Row]]:
        column = "clearing" if meta.mode == "whole" else "cleared"

        def read(f: Path) -> list[Row]:
            df = pd.read_parquet(f)
            audit(df.columns, GREEDY, f)
            other = "clearing" if column == "cleared" else "cleared"
            if column not in df.columns or other in df.columns:
                raise ValueError(f"{f}: a run holds one kind of row (mode {meta.mode})")
            out: list[Row] = []
            for _, r in df.iterrows():
                if len(r[column]) == 0:
                    continue                # the D 0 row
                row = commit(r, GREEDY)
                row.update(block=r["block"], n=int(r["n"]))
                row.setdefault("step", len(out) + 1)
                out.append(row)
            if not out:
                raise ValueError(f"{f}: no row clears anything")
            same("block", df["block"].iloc[0], block_of(f), f)
            return out
        return read


# -- priority-weighted P (pweight.py: pweight_rows/<metric>/<simp|fw|score>/)

PWEIGHT_NOT_CLEARINGS: Mapping[str, str] = {
    "fw": "Frank-Wolfe bounds (numbers; the design x is not stored), no clearing",
    "score": "the clearings scored under every objective: scores only",
}


@dataclass(frozen=True)
class PweightFamily:
    def sources(self, root: Path) -> list[Source]:
        out: list[Source] = []
        for md in _metric_dirs(root, "pweight_rows"):
            for d in sorted((md / "simp").glob("*")):
                files = tuple(sorted(d.glob("*.parquet")))
                if not files:
                    continue
                first = pd.read_parquet(files[0], columns=["power", "demand", "plan", "along",
                                                           "budget", "objective"]).iloc[0]
                same("objective", first["objective"], d.name, files[0])
                along, search = split_along(str(first["along"]))
                same("metric", str(first["along"]), md.name, files[0])
                rel = d.relative_to(root).as_posix()
                meta = Settings(
                    run=rel, family="simp", mode="whole", spec=str(first["plan"]),
                    power=float(first["power"]), demand=str(first["demand"]), population="area",
                    along=along, search_along=search, mesh="", budget=float(first["budget"]),
                    source=rel,
                    note=f"priority-weighted P, objective {d.name} (pweight.py), SIMP base with "
                         "free first, on the CPU")
                out.append(Source(settings=meta, files=files,
                                  block_ids=tuple(block_of(f) for f in files),
                                  read=simp_reader(SIMP, meta)))
        return out


# -- rows a cluster run was shipped, since overwritten (the cluster launcher's input tars)

STORE_MEMBER = re.compile(r"^research/roadless/(?P<path>(?P<store>relax_rows|polish_rows|"
                          r"pweight_rows|clear_rows_[^/]+)/.+\.parquet)$")


@dataclass(frozen=True)
class ShippedFamily:
    """Every row under a clearing store in a cluster run's input tars (the road-b run shipped
    each large block's stored clearings, 2026-10-11 16:23) whose bytes differ from the local
    file's, or whose local file is gone, taken from the earliest run that shipped those bytes and
    extracted under `scratch`. The families above read them from there, unchanged; a source's run
    is theirs plus @<cluster run>. Rows identical to the local ones are the local ones. Rows under
    a metric not this code's (`_record`: the retired raster's, or another code's complex) are
    records: counted, not read."""
    cluster_runs: Path
    scratch: Path

    def sources(self, root: Path) -> list[Source]:
        tars = sorted(self.cluster_runs.glob("*/remote/inputs/*.tar"),
                      key=lambda p: (p.stat().st_mtime_ns, p.parts[-4], p.name))
        taken: set[tuple[str, str]] = set()
        shipped: dict[str, int] = defaultdict(int)
        records: dict[str, int] = defaultdict(int)
        for tar in tars:
            cluster_run = tar.parts[-4]
            with tarfile.open(tar) as tf:
                for m in tf.getmembers():
                    hit = STORE_MEMBER.fullmatch(m.name) if m.isfile() else None
                    if hit is None:
                        continue
                    if _record(hit["path"]):
                        records[cluster_run] += 1
                        continue
                    fh = tf.extractfile(m)
                    if fh is None:
                        raise ValueError(f"{tar}: {m.name} is not a regular file")
                    data = fh.read()
                    local = root / hit["path"]
                    if local.exists() and local.read_bytes() == data:
                        continue
                    key = (hit["path"], hashlib.sha256(data).hexdigest())
                    if key in taken:
                        continue
                    taken.add(key)
                    dest = self.scratch / cluster_run / hit["path"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    os.utime(dest, (m.mtime, m.mtime))
                    shipped[cluster_run] += 1
        if records:
            print(f"shipped rows under a metric not this code's (records), not read: "
                  f"{sum(records.values())} in {len(records)} cluster runs", flush=True)
        out: list[Source] = []
        for cluster_run in sorted(shipped):
            sub = self.scratch / cluster_run
            for fam in FAMILIES:
                for src in fam.sources(sub):
                    meta = src.settings
                    out.append(replace(src, settings=replace(
                        meta, run=f"{meta.run}@{cluster_run}",
                        source=f".cluster_submit/runs/{cluster_run}/remote/inputs/*.tar"
                               f"[{meta.source}]",
                        note=f"the rows as run {cluster_run} shipped them ({len(src.files)} "
                             "blocks' input tars), which differ from the local rows now: "
                             "overwritten in place since")))
        return out


def _record(path: str) -> bool:
    """A rows path (a store's, under research/roadless) under a metric not this code's."""
    parts = path.split("/")
    if parts[0].startswith("clear_rows_"):
        m = GREEDY_DIR.fullmatch(parts[0])
        return m is None or not this_code(m["along"])
    return len(parts) < 2 or not this_code(parts[1])


FAMILIES: tuple[Family, ...] = (SimpFamily(), PolishFamily(), GreedyFamily(), PweightFamily())


ROOTS = ("relax_rows", "polish_rows", "pweight_rows")
CLUSTER_RUNS = HERE.parent.parent / ".cluster_submit" / "runs"      # the repo root's launcher dir


def records(root: Path) -> list[str]:
    """The rows directories the sources do not read (records: `this_code`): each store's metric
    directories under another metric, and the greedy-format directories that are not this code's
    (clear.rows_dir's)."""
    out = [d.relative_to(root).as_posix() for r in ROOTS if (root / r).is_dir()
           for d in sorted((root / r).iterdir()) if d.is_dir() and not this_code(d.name)]
    return out + [d.name for d in sorted(root.glob("clear_rows_*"))
                  if d.is_dir() and _record(d.name + "/x.parquet")]


def discover(root: Path, shipped: ShippedFamily) -> list[Source]:
    """Every run the sources hold, with the closed-set audit: every directory holding parquets
    under this code's metric in a store is exported or not clearings (pweight's fw and score),
    every top-level directory that looks like a rows store is a store's or in EXCLUDED, and every
    top-level parquet with a clearing list is a known record."""
    found = [s for fam in (*FAMILIES, shipped) for s in fam.sources(root)]
    exported = {(root / s.settings.source) for s in found if "[" not in s.settings.source}
    live = [md for r in ROOTS for md in _metric_dirs(root, r)] + [
        d for d in sorted(root.glob("clear_rows_*")) if d.is_dir()
        and not _record(d.name + "/x.parquet")]
    for store in live:
        for d in sorted({f.parent for f in store.rglob("*.parquet")}):
            rel = d.relative_to(root).as_posix()
            sub = d.relative_to(store).parts
            not_clearings = store.parent.name == "pweight_rows" and sub[0] in PWEIGHT_NOT_CLEARINGS
            if d not in exported and not not_clearings:
                raise ValueError(f"{rel}: holds parquets under this code's metric, is not exported")
    for d in sorted(root.iterdir()):
        looks = d.is_dir() and (d.name.endswith("_rows") or "_rows_" in d.name
                                or d.name.startswith(("rows_", "clear_rows_")))
        if looks and d.name not in ROOTS and not d.name.startswith("clear_rows_") \
                and d.name not in EXCLUDED:
            raise ValueError(f"{d.name}: a rows store that is neither a store's nor in EXCLUDED")
    for f in sorted(root.glob("*.parquet")):
        if f.name in RECORD_PARQUETS:
            continue
        if any(fld.name in ID_COLUMNS and pa.types.is_list(fld.type) for fld in pq.read_schema(f)):
            raise ValueError(f"{f.name}: holds clearings (a list column) but is neither exported "
                             "nor a known record")
    return found


# ------------------------------------------------------------------ blocks

@dataclass(frozen=True)
class BlockData:
    """A block as the export needs it: its print and its footprints' areas."""
    print: BlockPrint
    areas: np.ndarray

    def cost(self, population: str) -> np.ndarray:
        """Each building's share of the population (common.POPULATIONS: area share or 1 / n)."""
        if population == "area":
            return self.areas / self.areas.sum()
        if population == "count":
            return np.full(self.print.n, 1.0 / self.print.n)
        raise ValueError(f"unknown population {population!r}")


def build_blocks(block_ids: Iterable[str], *, batch: int) -> dict[str, BlockData]:
    """The blocks through common.build_blocks, `batch` at a time (a region's build is the long one),
    kept as their prints and areas."""
    import common
    import shapely
    todo = sorted(set(block_ids))
    out: dict[str, BlockData] = {}
    for k in range(0, len(todo), batch):
        for b in common.build_blocks(todo[k:k + batch]):
            polys = np.asarray(b.buildings.outlines)
            out[b.block_id] = BlockData(print=BlockPrint.of(b), areas=shapely.area(polys))
        print(f"  blocks {min(k + batch, len(todo))}/{len(todo)}", flush=True)
    missing = set(todo) - set(out)
    if missing:
        raise ValueError(f"build_blocks did not return {sorted(missing)}")
    return out


# ------------------------------------------------------------------ one run

def local_time(t: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t).astimezone()


def stamp(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M")


def stated(r: Mapping[str, Any], column: str) -> bool:
    """Whether a row states a real: the column is there and not blank (a committed row's blank is
    NaN)."""
    return column in r and not pd.isna(r[column])


def check_rows(run: str, mode: str, population: str, rows: list[Row],
               data: Mapping[str, BlockData]) -> None:
    """The checks of one run's rows against its blocks; fills in D where the row states none (the
    collected sets: the share of the positions)."""
    steps = clearings.steps_of(mode)
    by_block: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_block[str(r["block"])].append(r)
    for block, rs in by_block.items():
        bd = data[block]
        cost = bd.cost(population)
        rs.sort(key=lambda r: int(r["step"]))
        if [int(r["step"]) for r in rs] != list(range(1, len(rs) + 1)):
            raise ValueError(f"{run} {block}: steps {[r['step'] for r in rs]}")
        parts: list[np.ndarray] = []
        for r in rs:
            where = f"{run} {block} step {r['step']}"
            if stated(r, "n") and int(r["n"]) != bd.print.n:
                raise ValueError(f"{where}: the row says {r['n']} buildings, the block has "
                                 f"{bd.print.n}")
            parts.append(np.asarray(r["cleared"], dtype=np.int64))
            through = steps.through(parts, len(parts))
            if len(through) and (through.min() < 0 or through.max() >= bd.print.n):
                raise ValueError(f"{where}: positions outside 0..{bd.print.n - 1}")
            if len(np.unique(through)) != len(through):
                raise ValueError(f"{where}: a building is cleared twice")
            d = float(cost[through].sum())
            if stated(r, "D"):
                if abs(float(r["D"]) - d) > D_TOL:
                    raise ValueError(f"{where}: D {r['D']} but its positions' share is {d}")
            else:
                r["D"] = d
            if stated(r, "free_D"):
                f = np.asarray(r["free_cleared"], dtype=np.int64)
                if abs(float(r["free_D"]) - float(cost[f].sum())) > D_TOL:
                    raise ValueError(f"{where}: free_D {r['free_D']} but the free buildings' share "
                                     f"is {cost[f].sum()}")
                if not np.isin(f, through).all():
                    raise ValueError(f"{where}: free first's buildings are not all cleared")


def frame_of(rows: Sequence[Row]) -> pd.DataFrame:
    """The rows without `n`, as the columns the store writes."""
    return pd.DataFrame([{k: v for k, v in r.items() if k != "n"} for r in rows])


def read_run(src: Source, data: Mapping[str, BlockData]) -> tuple[pd.DataFrame, RunInfo]:
    """Read, check and describe one run."""
    meta = src.settings
    rows: list[Row] = []
    mtimes = [local_time(f.stat().st_mtime) for f in src.files]
    for f in src.files:
        got = src.read(f)
        rows += got
        if not all(stated(r, "enclosed") for r in got):
            raise ValueError(f"{f}: a row with no `enclosed`: B (i) unstated (every writer of "
                             "this code's metric states it)")
    if len(set(src.block_ids)) != len(src.block_ids) and len(src.files) > 1:
        raise ValueError(f"{meta.run}: two files for one block")
    check_rows(meta.run, meta.mode, meta.population, rows, data)
    metric = "B"
    info = RunInfo(
        run=meta.run, family=meta.family, mode=meta.mode, spec=meta.spec, power=meta.power,
        demand=meta.demand, population=meta.population, along=meta.along,
        search_along=meta.search_along, mesh=meta.mesh, budget=meta.budget, source=meta.source,
        source_files=len(src.files), source_bytes=sum(f.stat().st_size for f in src.files),
        written_first=stamp(min(mtimes)), written_last=stamp(max(mtimes)), metric=metric,
        blocks=len({str(r["block"]) for r in rows}), rows=len(rows),
        ids=sum(len(r["cleared"]) for r in rows), note=meta.note)
    return frame_of(rows), info


# ------------------------------------------------------------------ merging with what is committed

def row_texts(frame: pd.DataFrame) -> dict[str, list[tuple[str, ...]]]:
    """Each block's rows as the text a file would hold of every column (blank where the frame has
    none), in step order: what two rows must share to be the same row."""
    cols = clearings.COLUMNS
    out: dict[str, list[tuple[int, tuple[str, ...]]]] = defaultdict(list)
    for i in range(len(frame)):
        cells = tuple(c.text(frame[c.name].iloc[i]) if c.name in frame.columns else ""
                      for c in cols)
        out[str(frame["block"].iloc[i])].append((int(frame["step"].iloc[i]), cells))
    return {b: [cells for _, cells in sorted(rs)] for b, rs in out.items()}


@dataclass(frozen=True)
class Merged:
    frame: pd.DataFrame
    added: int          # blocks new to the store
    kept: int           # blocks only the store holds
    changed: list[str]  # blocks whose rows differ from the store's


def merge(store: Store, run: str, new: pd.DataFrame) -> Merged:
    if not store.path(run).exists():
        return Merged(new, added=new["block"].nunique(), kept=0, changed=[])
    old = store.table(run)
    tn, to = row_texts(new), row_texts(old)
    changed = sorted(b for b in tn.keys() & to.keys() if tn[b] != to[b])
    only_old = sorted(to.keys() - tn.keys())
    gone = set(only_old)
    keep = pd.DataFrame(old.iloc[[i for i, b in enumerate(old["block"]) if b in gone]])
    frame = pd.concat([new, keep], ignore_index=True) if len(keep) else new
    return Merged(frame, added=len(tn.keys() - to.keys()), kept=len(only_old), changed=changed)


def merged_info(old: RunInfo | None, info: RunInfo, frame: pd.DataFrame) -> RunInfo:
    """`info` with the statistics of the merged frame and the writing times of both."""
    first, last = info.written_first, info.written_last
    metric = info.metric
    if old is not None:
        first, last = min(first, old.written_first), max(last, old.written_last)
        metric = metric if metric == old.metric else "mixed"
    return replace(info, written_first=first, written_last=last, metric=metric,
                   blocks=int(frame["block"].nunique()), rows=len(frame),
                   ids=int(sum(len(c) for c in frame["cleared"])))


# ------------------------------------------------------------------ the commands

def inventory(root: Path) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        _inventory(root, discover(root, ShippedFamily(CLUSTER_RUNS, Path(scratch))))


def _inventory(root: Path, src: list[Source]) -> None:
    print(f"{'family':10s} {'p':>3s} {'pop':5s} {'D':>5s} {'files':>5s} {'blocks':>6s} "
          f"{'KB':>6s}  {'written (first .. last)':35s} along  run")
    for s in src:
        m = s.settings
        mt = [local_time(f.stat().st_mtime) for f in s.files]
        along = (m.along or "-") + (f"@{m.search_along}" if m.search_along else "")
        print(f"{m.family:10s} {m.power if m.power else '-'!s:>3.3s} {m.population:5s} "
              f"{m.budget if m.budget else '-'!s:>5.5s} {len(s.files):5d} "
              f"{len(set(s.block_ids)):6d} {sum(f.stat().st_size for f in s.files) / 1e3:6.0f}  "
              f"{stamp(min(mt))} .. {stamp(max(mt))}  {along}  {m.run}")
    print(f"{len(src)} runs; {sum(len(s.files) for s in src)} row files, "
          f"{sum(f.stat().st_size for s in src for f in s.files) / 1e6:.1f} MB")
    rec = records(root)
    print(f"{len(rec)} rows directories under a metric not this code's: records, not read "
          f"({', '.join(rec[:4])}{', ...' if len(rec) > 4 else ''}); "
          f"{len(RECORD_PARQUETS)} top-level parquets of clearings, records")


def export(root: Path, store: Store, replace_runs: set[str]) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        sources = discover(root, ShippedFamily(CLUSTER_RUNS, Path(scratch)))
        _export(root, store, replace_runs, sources)


def _export(root: Path, store: Store, replace_runs: set[str], sources: list[Source]) -> None:
    unknown = replace_runs - {s.settings.run for s in sources}
    if unknown:
        raise ValueError(f"--replace names no run: {sorted(unknown)}")
    print(f"{len(sources)} runs; building the blocks they name", flush=True)
    data = build_blocks({b for s in sources for b in s.block_ids}, batch=40)

    committed = store.prints() if (store.root / clearings.BLOCKS).exists() else {}
    for block, bd in data.items():
        old = committed.get(block)
        if old is not None and old != bd.print:
            raise clearings.StaleClearing(
                f"block {block}: its footprints changed since it was first exported (n {old.n} -> "
                f"{bd.print.n}, fingerprint {old.fingerprint[:12]} -> {bd.print.fingerprint[:12]}): "
                "its committed clearings no longer index its buildings; migrate them, do not "
                "re-bless them")

    old_runs = store.runs() if (store.root / clearings.RUNS).exists() else {}
    frames: dict[str, pd.DataFrame] = {}
    infos: dict[str, RunInfo] = {}
    changed: dict[str, list[str]] = {}
    for k, src in enumerate(sources, 1):
        run = src.settings.run
        new, info = read_run(src, data)
        m = merge(store, run, new)
        if m.changed and run not in replace_runs:
            changed[run] = m.changed
            continue
        frames[run], infos[run] = m.frame, merged_info(old_runs.get(run), info, m.frame)
        if k % 20 == 0:
            print(f"  read {k}/{len(sources)} runs", flush=True)
    if changed:
        lines = [f"  {run}: {len(bl)} blocks {bl[:4]}{'...' if len(bl) > 4 else ''}"
                 for run, bl in changed.items()]
        raise SystemExit("rows the sources hold differ from the committed ones (an overwrite in "
                         "place; the committed rows stand): re-run with --replace RUN to take "
                         "the new ones (the old stay in git's history)\n" + "\n".join(lines))

    store.write_prints({**committed, **{b: d.print for b, d in data.items()}}.values())
    for run, frame in frames.items():
        store.write_table(run, frame)
    store.write_runs({**old_runs, **infos}.values())          # the index last: it lists what is whole

    files = [p for p in store.root.rglob("*.csv")]
    raw = sum(p.stat().st_size for p in files)
    packed = sum(len(zlib.compress(p.read_bytes(), 6)) for p in files)
    print(f"wrote {len(frames)} runs, {sum(len(f) for f in frames.values())} rows, "
          f"{sum(i.ids for i in infos.values())} positions, {len(data)} blocks; "
          f"{len(files)} files, {raw / 1e6:.2f} MB, {packed / 1e6:.2f} MB zlib'd (what git stores)")


def main(argv: Sequence[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory")
    ex = sub.add_parser("export")
    ex.add_argument("--replace", default="", help="runs whose differing rows the sources win")
    ex.add_argument("--store", type=Path, default=clearings.COMMITTED,
                    help="where to write (default: the committed store)")
    args = ap.parse_args(argv)
    if args.cmd == "inventory":
        inventory(HERE)
    else:
        export(HERE, Store(args.store), {r for r in args.replace.split(",") if r})


if __name__ == "__main__":
    main(sys.argv[1:])
