"""Export the stored clearings' core into research/roadless/clearings/ (format: clearings.py).

    PYTHONPATH=. uv run python research/roadless/clearings_export.py inventory
    PYTHONPATH=. uv run python research/roadless/clearings_export.py export [--replace RUN[,RUN...]]

From the repo root, on the CPU, single-threaded (OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1). `inventory` lists what the sources hold and writes nothing.
`export` reads every source run, builds the blocks they name (common.build_blocks), checks each row
against its block (the building count the row recorded; positions in range and distinct; D
recomputed from the positions equal to the row's own D) and against the run's settings (the path's
against the row's own columns), and writes the committed core.

The sources, each a Family (a Strategy resolved here, where the directories are named) that finds
its runs, parses the settings their names encode and reads one file into core rows:

    SimpFamily       relax_rows/<along>/<plan>/D<budget>/<block>_p<power>.parquet: SIMP's clearing
                     (simp_cleared; run <dir>.p<power>) and, in the Frank-Wolfe plans' rows, the
                     rounding of the relaxation (cleared; run <dir>.p<power>.fw)
    PolishFamily     polish_rows/<along>/<picker>.P<tries>w<width>[x2]<mesh>/D<budget>/..._p<power>
    GreedyFamily     clear_rows_<picker>_h<h>[_area][_p<power>][_<along>[@<search>]]/<block>.parquet:
                     the greedy and the searches that write its format (nested rows `cleared`,
                     or a whole clearing a row `clearing`)
    PweightFamily    pweight_rows/simp/<objective>/<block>.parquet (priority-weighted P)
    CollectedFamily  the clearings the studies gathered into one parquet (fine_clearings, ...): a
                     run for each (file, name), whose note says how many of its sets equal a set
                     of a stored run
    ShippedFamily    rows of the stores above that a cluster run was shipped in its input tars
                     (.cluster_submit/runs/<run>/remote/inputs/<block>.tar) and that differ from
                     the local row now or are gone from it: the rows a later run overwrote in
                     place, read by the family of their store as run <run id>@<cluster run>

Every source column is committed, checked, or deliberately dropped with a reason (a Slot); one that
is none of them raises, so a column a new optimizer adds is never silently left out. So does a
directory under research/roadless that looks like a rows store and is neither exported nor in
EXCLUDED with its reason.

relax220_p1.parquet and relax220_p2.parquet (COPIES) copy the rows of a Frank-Wolfe run and are
checked equal to its two committed runs, not exported again. Rows are read without SIMP's design x
(a float a building, half the bytes of the rows, not the clearing).

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

# lifted.Placement (7913fe4, 2026-10-11 16:21:16 -0400): the scorer from which a home with no ring
# cell injects on its open side. A row written before it scored without those homes; one with
# `enclosed` was made after.
B_EPOCH = dt.datetime(2026, 10, 11, 16, 21, 16, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
D_TOL = 1e-9        # a row's D against the one recomputed from its positions (sums in another order)

NUM = r"[0-9]+(?:\.[0-9]+)?"
# A run's metric, as its rows' directory names it: a raster conductance (lifted.conductance_of,
# `<along>[+<cells>]+apwfs`; a bare along-conductance, lifted.along_of, the step rule's before the
# face aperture, 2026-10-13), with the search's after `@` where it differed; or the gap complex's
# (gap_clear.complex_of: `cx+...`, its code's digest in it, read only if it is this code's)
ALONG = rf"(?:uni|sl{NUM}|ss{NUM}k{NUM}(?:n{NUM}r{NUM})?t?)(?:\+[bd]{NUM})?(?:\+apwfs)?"
ALONGS = rf"(?:{ALONG}(?:@{ALONG})?|cx\+[^/@]+(?:@e{NUM})?)"
MESH_TAIL = re.compile(rf"^(?P<base>.*?)(?:\.h(?P<h>{NUM}))?"
                       rf"(?P<local>\.[ag]{NUM}x{NUM}(?:f{NUM}p{NUM}o{NUM})?)?$")

# Directories that look like rows stores and are not clearings, and the file-level equivalents.
EXCLUDED: Mapping[str, str] = {
    "insulator_rows": "scenario scores of the insulator world; the clearings it scored are "
                      "insulator_rows/walls_clearings.parquet, collected here",
    "road_rows": "road networks and their scores (BACKLOG 4), no clearing",
    "transparency_rows": "road networks and their scores (BACKLOG 5), no clearing",
    "res_rows": "mesh-resolution rescoring of stored clearings: scores only",
    "rows_h0.5_ell3_K8": "the 220-block study's road-network scores, no clearing",
    "rows_h0.5_ell3_K8_area": "the 220-block study's road-network scores, no clearing",
    "rows_h0.5_ell3_K8_area_p1only": "the 220-block study's road-network scores, no clearing",
    "pweight_rows/fw": "Frank-Wolfe bounds (numbers; the design x is not stored), no clearing",
    "pweight_rows/score": "the clearings scored under every objective: scores only",
    "aperture_rows": "the grid fix's study (NOTES, 'The grid fix: steps that conduct by their "
                     "open width'): scores under the step rules it compared, and SIMP's clearing "
                     "under Fws on one block (simp/Fws), a study's record (summary_*.csv)",
    "width_rows": "the effective-width study (NOTES, 'Uniform conductance with an effective "
                  "width'): scores, and SIMP's clearings under B_r and D_s on the sentinel "
                  "(simp/<variant>), a study's record (summary_*.csv)",
    "cross_rows": "the cross-checks of the complex against the raster (crosscheck.py, phase 2b): "
                  "the raster's scores of the road prefixes and stored clearings, no clearing",
    "gap_rows": "the gap complex's study and the runtime's calibration (NOTES, 'The gap complex' "
                "and 'The complex as the runtime'): scores; simp/ the width forms' SIMP clearings "
                "on the rebuilt complex (a study's record); runtime/preset/ the acceptance's "
                "per-device copies of the complex presets' rows (the same clearings as "
                "relax_rows/<metric> and polish_rows/<metric>, exported) and the dominated "
                "+relaxlumped's (deleted; its numbers in NOTES)",
}
# Top-level parquets that copy rows of a run, checked equal to them at export.
COPIES: Mapping[str, str] = {
    "relax220_p1.parquet": "relax_rows/uni/fw40.q3.i20.t1e-05/D0.1.p1",
    "relax220_p2.parquet": "relax_rows/uni/fw40.q3.i20.t1e-05/D0.1.p2",
}
# Source columns of the clearing lists: a top-level parquet holding one is exported or in COPIES.
ID_COLUMNS = ("cleared", "simp_cleared", "clearing")


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


def mesh_name(spec: str) -> str:
    """lifted.MeshSpec.name of a plan's or a polish spec's mesh suffix (.h<h> and/or the local
    or goal mesh; none: h 0.5)."""
    m = MESH_TAIL.fullmatch(spec)
    assert m is not None
    return f"h{m['h'] or '0.5'}" + (m["local"][1:] if m["local"] else "")


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
          "device": "the solves' device (seam.Metric.device): scores alike to the floating "
                    "point, not the run's identity"},
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
    """(the run's metric, the search's where it differed: a raster's `@<along>`, the complex's
    `@e<eps>`, the translucent search's eps world). A complex metric's name is checked whole
    (gap_clear.complex_of: refused unless it spells a metric of this code's constants), so no other
    code's rows are read as this code's."""
    if token.startswith(metrics.COMPLEX):
        gap_clear.complex_of(token, factorization=gap_factor.SuperLU())
    metric, _, search = token.partition("@")
    return metric, search


def mesh_of_run(spec: str, along: str) -> str:
    """The grid a run optimized and scored on (mesh_name), none for the complex's ('')."""
    return "" if along.startswith(metrics.COMPLEX) else mesh_name(spec)


@dataclass(frozen=True)
class SimpFamily:
    def sources(self, root: Path) -> list[Source]:
        out: list[Source] = []
        pat = re.compile(rf"^relax_rows/(?P<along>{ALONGS})/(?P<plan>[^/]+)/D(?P<budget>{NUM})$")
        for d in sorted((root / "relax_rows").glob("*/*/D*")):
            rel = d.relative_to(root).as_posix()
            m = pat.fullmatch(rel)
            if m is None:
                raise ValueError(f"{rel}: not relax_rows/<along>/<plan>/D<budget>")
            by_power: dict[str, list[Path]] = defaultdict(list)
            for f in sorted(d.glob("*.parquet")):
                p = re.fullmatch(r".+_p(?P<power>[0-9.]+)", f.stem)
                if p is None:
                    raise ValueError(f"{f}: not <block>_p<power>.parquet")
                by_power[p["power"]].append(f)
            along, search = split_along(m["along"])
            for power, files in by_power.items():
                base = Settings(run=f"{rel}.p{power}", family="simp", mode="whole", spec=m["plan"],
                                power=float(power), demand="w", population="area", along=along,
                                search_along=search, mesh=mesh_of_run(m["plan"], along),
                                budget=float(m["budget"]), source=rel, note="")
                out.append(Source(settings=base, files=tuple(files),
                                  block_ids=tuple(block_of(f) for f in files),
                                  read=simp_reader(SIMP, base)))
                fw_files = tuple(f for f in files
                                 if "rounded_perm" in pq.read_schema(f).names)
                if fw_files:
                    fw = replace(base, run=f"{rel}.p{power}.fw", family="simp-fw",
                                 note="the relaxation's rounding (round_by_x), not SIMP's answer")
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
    check=("block", "n", "picker", "mesh"),
    drop={"device": SIMP.drop["device"]})


@dataclass(frozen=True)
class PolishFamily:
    def sources(self, root: Path) -> list[Source]:
        out: list[Source] = []
        pat = re.compile(rf"^polish_rows/(?P<along>{ALONGS})/(?P<spec>(?P<picker>S{NUM}cat"
                         rf"(?:w[0-9]+)?)\.P[0-9]+w[0-9]+(?:x2)?[^/]*)/D(?P<budget>{NUM})$")
        for d in sorted((root / "polish_rows").glob("*/*/D*")):
            rel = d.relative_to(root).as_posix()
            m = pat.fullmatch(rel)
            if m is None:
                raise ValueError(f"{rel}: not polish_rows/<along>/<picker>.P<t>w<w>[x2]/D<budget>")
            by_power: dict[str, list[Path]] = defaultdict(list)
            for f in sorted(d.glob("*.parquet")):
                p = re.fullmatch(r".+_p(?P<power>[0-9.]+)", f.stem)
                if p is None:
                    raise ValueError(f"{f}: not <block>_p<power>.parquet")
                by_power[p["power"]].append(f)
            along, search = split_along(m["along"])
            for power, files in by_power.items():
                meta = Settings(run=f"{rel}.p{power}", family="polish", mode="whole",
                                spec=m["spec"], power=float(power), demand="w",
                                population="area", along=along, search_along=search,
                                mesh=mesh_of_run(m["spec"], along), budget=float(m["budget"]),
                                source=rel, note="")
                out.append(Source(settings=meta, files=tuple(files),
                                  block_ids=tuple(block_of(f) for f in files),
                                  read=self._reader(meta, m["picker"])))
        return out

    @staticmethod
    def _reader(meta: Settings, picker: str) -> Callable[[Path], list[Row]]:
        def read(f: Path) -> list[Row]:
            df = pd.read_parquet(f)
            audit(df.columns, POLISH, f)
            [r] = (row for _, row in df.iterrows())
            same("block", r["block"], block_of(f), f)
            same("picker", r["picker"], picker, f)
            # the row's own `mesh` is the grid its spec names (the complex's: a placeholder it
            # ignores, h 0.5 unnamed)
            same("mesh", r["mesh"], mesh_name(meta.spec), f)
            row = commit(r, POLISH)
            row.update(block=r["block"], step=1, n=int(r["n"]))
            return [row]
        return read


# -- the greedy and the searches that write its format (clear.greedy_block, search.py)

GREEDY = Slot(
    commit={"step": "step", "D": "D", "perm": "perm", "perm1": "perm1", "cleared": "cleared",
            "clearing": "cleared", "P0": "P0", "t": "t"},
    check=("block", "n"),
    drop={})
GREEDY_DIR = re.compile(rf"^clear_rows_(?P<picker>.+?)_h(?P<h>{NUM})(?:_(?P<pop>area))?"
                        rf"(?:_p(?P<power>{NUM}))?(?:_(?P<along>{ALONGS}))?"
                        rf"(?P<trapped>\.trapped)?$")


def greedy_family(picker: str) -> str:
    if picker.startswith("SIMP"):
        return "simp-swing" if "+" in picker else "simp-path"
    for prefix, family in (("GP", "grow-prune"), ("FL", "floating"), ("SW", "swing")):
        if picker.startswith(prefix):
            return family
    if re.match(r"^(M|B|S)[0-9]", picker):
        return "greedy"
    raise ValueError(f"unknown picker {picker!r}")


@dataclass(frozen=True)
class GreedyFamily:
    def sources(self, root: Path) -> list[Source]:
        out: list[Source] = []
        for d in sorted(root.glob("clear_rows_*")):
            if not d.is_dir():
                continue
            m = GREEDY_DIR.fullmatch(d.name)
            if m is None:
                raise ValueError(f"{d.name}: not clear.rows_dir's name")
            files = tuple(sorted(d.glob("*.parquet")))
            if not files:
                continue                    # an empty directory (a run that wrote nothing)
            names = pq.read_schema(files[0]).names
            mode = "whole" if "clearing" in names else "added"
            family = greedy_family(m["picker"])
            along, search = split_along(m["along"] or "uni")
            budget = re.search(rf"D({NUM})$", m["picker"])
            meta = Settings(
                run=d.name, family=family, mode=mode, spec=m["picker"],
                power=float(m["power"] or 1.0), demand="w", population=m["pop"] or "count",
                along=along, search_along=search, mesh=f"h{m['h']}",
                budget=float(budget[1]) if budget and family != "greedy" else None,
                source=d.name,
                note="a renamed run: SIMP's cold nested path, whose switched-off buildings stay "
                     "trapped (relax.path)" if m["trapped"] else "")
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


# -- priority-weighted P (pweight.py)

@dataclass(frozen=True)
class PweightFamily:
    def sources(self, root: Path) -> list[Source]:
        out: list[Source] = []
        for d in sorted((root / "pweight_rows" / "simp").glob("*")):
            files = tuple(sorted(d.glob("*.parquet")))
            if not files:
                continue
            first = pd.read_parquet(files[0], columns=["power", "demand", "plan", "along",
                                                       "budget", "objective"]).iloc[0]
            same("objective", first["objective"], d.name, files[0])
            meta = Settings(
                run=f"pweight_rows/simp/{d.name}", family="simp", mode="whole",
                spec=str(first["plan"]), power=float(first["power"]), demand=str(first["demand"]),
                population="area", along=str(first["along"]), search_along="",
                mesh=mesh_name(str(first["plan"])), budget=float(first["budget"]),
                source=d.relative_to(root).as_posix(),
                note=f"priority-weighted P, objective {d.name} (pweight.py), SIMP base with "
                     "free first, on the CPU")
            out.append(Source(settings=meta, files=files,
                              block_ids=tuple(block_of(f) for f in files),
                              read=simp_reader(SIMP, meta)))
        return out


# -- the clearings the studies gathered (one parquet, a name each)

COLLECTED: Mapping[str, str] = {
    "fine_clearings.parquet": "clearings gathered for resolution_check.py (its clearings.parquet "
                              "input; runs fine25, fine35), 2026-10-10",
    "cheap_a5_clearings.parquet": "the cheap preset's clearings on the composite mesh a5x8, "
                                  "gathered for resolution_check.py, 2026-10-08",
    "ov_clearings.parquet": "the oversized blocks' coarse-h runs' clearings, kept when those runs "
                            "were superseded (NOTES, 'Local coarsening: built'); perm is Lens A "
                            "on the block's own mesh `h`; 2026-10-08",
    "ovgoal_clearings.parquet": "the oversized blocks' clearings gathered for resolution_check.py "
                                "on goal meshes, 2026-10-08",
    "res_clearings.parquet": "the five clearings of each of the 59 large blocks gathered for "
                             "resolution_check.py (run res-mesh; NOTES, 'Local coarsening: "
                             "built'); greedy_steps is a greedy's whole pick order, 2026-10-08",
    "three_clearings.parquet": "14401, 32841 and 7851's clearings gathered for resolution_check.py "
                               "(runs three-res, three-res2; NOTES, 'Local coarsening: built'), "
                               "2026-10-08",
    "insulator_rows/walls_clearings.parquet": "the walls study's stored clearings (SIMP's default "
                                              "and the cheap preset on 16 large blocks) as they "
                                              "stood before lifted.Placement, 2026-10-11 12:50; "
                                              "perm is their stored Lens A of that day",
}
COLLECTED_PERM = {"stored_perm": "perm", "perm_h": "perm"}


@dataclass(frozen=True)
class CollectedFamily:
    def sources(self, root: Path) -> list[Source]:
        out: list[Source] = []
        for rel, what in COLLECTED.items():
            f = root / rel
            df = pd.read_parquet(f)
            extra = sorted(set(df.columns) - {"block", "name", "cleared", "h", "perm_h",
                                              "stored_perm"})
            if extra:
                raise ValueError(f"{f}: columns {extra} are not classified")
            for name in sorted(df["name"].unique()):
                meta = Settings(
                    run=f"collected/{rel.removesuffix('.parquet')}/{name}", family="collected",
                    mode="whole", spec=name, power=None, demand="w", population="area", along="",
                    search_along="", mesh="", budget=None, source=f"{rel}[name={name}]",
                    note=what)
                blocks = tuple(sorted(df.loc[df["name"] == name, "block"]))
                out.append(Source(settings=meta, files=(f,), block_ids=blocks,
                                  read=self._reader(meta, name)))
        return out

    @staticmethod
    def _reader(meta: Settings, name: str) -> Callable[[Path], list[Row]]:
        def read(f: Path) -> list[Row]:
            df = pd.read_parquet(f)
            out: list[Row] = []
            for _, r in df[df["name"] == name].iterrows():
                row: Row = dict(block=r["block"], step=1, cleared=ids(r["cleared"]))
                for src, core in COLLECTED_PERM.items():
                    if src in r.index:
                        row[core] = r[src]
                if "h" in r.index:
                    row["h"] = r["h"]
                out.append(row)
            return out
        return read


# -- rows a cluster run was shipped, since overwritten (the cluster launcher's input tars)

STORE_MEMBER = re.compile(r"^research/roadless/(?P<path>(?:relax_rows|polish_rows|pweight_rows|"
                          r"clear_rows_[^/]+)/.+\.parquet)$")


@dataclass(frozen=True)
class ShippedFamily:
    """Every row under a clearing store in a cluster run's input tars (the road-b run shipped
    each large block's stored clearings, 2026-10-11 16:23) whose bytes differ from the local
    file's, or whose local file is gone, taken from the earliest run that shipped those bytes and
    extracted under `scratch`. The families above read them from there, unchanged; a source's run
    is theirs plus @<cluster run>. Rows identical to the local ones are the local ones. Rows under
    a gap complex metric that is not this code's (`foreign`: phase 1's runs shipped to the raster's
    re-scoring, rcx-score*, before their migration to the digested name) are no rows of this code:
    the migration (gap_calib rescore) carried them, and they are counted, not read."""
    cluster_runs: Path
    scratch: Path

    def sources(self, root: Path) -> list[Source]:
        tars = sorted(self.cluster_runs.glob("*/remote/inputs/*.tar"),
                      key=lambda p: (p.stat().st_mtime_ns, p.parts[-4], p.name))
        taken: set[tuple[str, str]] = set()
        shipped: dict[str, int] = defaultdict(int)
        foreign: dict[str, int] = defaultdict(int)
        for tar in tars:
            cluster_run = tar.parts[-4]
            with tarfile.open(tar) as tf:
                for m in tf.getmembers():
                    hit = STORE_MEMBER.fullmatch(m.name) if m.isfile() else None
                    if hit is None:
                        continue
                    if _foreign(hit["path"]):
                        foreign[cluster_run] += 1
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
        if foreign:
            print(f"shipped rows under another code's complex metric, not read: "
                  f"{sum(foreign.values())} in {', '.join(sorted(foreign))}", flush=True)
        out: list[Source] = []
        for cluster_run in sorted(shipped):
            sub = self.scratch / cluster_run
            for fam in (SimpFamily(), PolishFamily(), GreedyFamily(), PweightFamily()):
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


def _foreign(path: str) -> bool:
    """A rows path under a gap complex metric whose name is not this code's (gap_clear.complex_of
    refuses it: another digest, or phase 1's undigested name)."""
    parts = path.split("/")
    if len(parts) < 2 or not parts[1].startswith(metrics.COMPLEX):
        return False
    try:
        gap_clear.complex_of(parts[1], factorization=gap_factor.SuperLU())
    except ValueError:
        return True
    return False


FAMILIES: tuple[Family, ...] = (SimpFamily(), PolishFamily(), GreedyFamily(), PweightFamily())


ROOTS = ("relax_rows", "polish_rows", "pweight_rows")
CLUSTER_RUNS = HERE.parent.parent / ".cluster_submit" / "runs"      # the repo root's launcher dir


def discover(root: Path, shipped: ShippedFamily) -> list[Source]:
    """Every run the sources hold, with the closed-set audit: every directory holding parquets
    under a store is exported or in EXCLUDED, and every top-level parquet with a clearing list is
    collected or a copy."""
    found = [s for fam in (*FAMILIES, CollectedFamily(), shipped) for s in fam.sources(root)]
    exported = {(root / s.settings.source) for s in found if "[" not in s.settings.source}

    def excluded(rel: str) -> bool:
        return any(rel == k or rel.startswith(k + "/") for k in EXCLUDED)

    stores = [root / r for r in ROOTS] + sorted(d for d in root.glob("clear_rows_*") if d.is_dir())
    for store in stores:
        for d in sorted({f.parent for f in store.rglob("*.parquet")}):
            rel = d.relative_to(root).as_posix()
            if d not in exported and not excluded(rel):
                raise ValueError(f"{rel}: holds parquets, is neither exported nor in EXCLUDED")
    for d in sorted(root.iterdir()):
        looks = d.is_dir() and (d.name.endswith("_rows") or "_rows_" in d.name
                                or d.name.startswith(("rows_", "clear_rows_")))
        if looks and d.name not in ROOTS and not d.name.startswith("clear_rows_") \
                and d.name not in EXCLUDED:
            raise ValueError(f"{d.name}: a rows store that is neither exported nor in EXCLUDED")
    for f in sorted(root.glob("*.parquet")):
        if f.name in COLLECTED or f.name in COPIES:
            continue
        if any(fld.name in ID_COLUMNS and pa.types.is_list(fld.type) for fld in pq.read_schema(f)):
            raise ValueError(f"{f.name}: holds clearings (a list column) but is neither collected "
                             "nor a copy of a run")
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
    epochs: list[str] = []
    mtimes = [local_time(f.stat().st_mtime) for f in src.files]
    for f, mt in zip(src.files, mtimes, strict=True):
        got = src.read(f)
        rows += got
        has = any("enclosed" in r and not pd.isna(r["enclosed"]) for r in got)
        if not has and mt >= B_EPOCH:
            raise ValueError(f"{f}: written {stamp(mt)}, after lifted.Placement, with no "
                             "`enclosed`: which scorer made it?")
        epochs.append("B" if has else "preB")
    if len(set(src.block_ids)) != len(src.block_ids) and len(src.files) > 1:
        raise ValueError(f"{meta.run}: two files for one block")
    check_rows(meta.run, meta.mode, meta.population, rows, data)
    metric = epochs[0] if len(set(epochs)) == 1 else "mixed"
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


# ------------------------------------------------------------------ copies and collected sets

def check_copies(root: Path, store_frames: Mapping[str, pd.DataFrame]) -> list[str]:
    """The top-level parquets that copy a run's rows (COPIES) equal to them: the same positions
    and the same lens a block, for SIMP's clearing and the rounding. Returns what it checked."""
    lines = []
    for name, run in COPIES.items():
        df = pd.read_parquet(root / name)
        for column, perm, target in (("simp_cleared", "simp_perm", run),
                                     ("cleared", "rounded_perm", run + ".fw")):
            t = store_frames[target].set_index("block")
            for _, r in df.iterrows():
                got = t.loc[r["block"]]
                if sorted(int(i) for i in r[column]) != sorted(int(i) for i in got["cleared"]):
                    raise ValueError(f"{name} {r['block']}: {column} differs from {target}")
                if abs(float(r[perm]) - float(got["perm"])) > 1e-12:
                    raise ValueError(f"{name} {r['block']}: {perm} differs from {target}")
        lines.append(f"{name}: {len(df)} blocks equal to {run} and {run}.fw")
    return lines


def collected_notes(frames: Mapping[str, pd.DataFrame], infos: Mapping[str, RunInfo]
                    ) -> dict[str, str]:
    """For each collected run, how many of its sets equal a set of a stored run (the same block,
    the same buildings): the collected copies' provenance, counted rather than assumed."""
    index: dict[tuple[str, frozenset[int]], int] = defaultdict(int)
    for run, frame in frames.items():
        info = infos[run]
        if info.family == "collected":
            continue
        steps = info.steps
        for block, g in frame.groupby("block", sort=False):
            rows = [np.asarray(c, dtype=np.int64) for c in g.sort_values("step")["cleared"]]
            for k in range(1, len(rows) + 1):
                index[(str(block), frozenset(int(i) for i in steps.through(rows, k)))] += 1
    out = {}
    for run, frame in frames.items():
        if infos[run].family != "collected":
            continue
        miss = [str(b) for b, c in zip(frame["block"], frame["cleared"], strict=True)
                if (str(b), frozenset(int(i) for i in c)) not in index]
        n = len(frame)
        out[run] = (f"{n - len(miss)} of {n} sets equal a set of a stored run"
                    + (f"; not elsewhere: {', '.join(sorted(miss))}" if miss else ""))
    return out


# ------------------------------------------------------------------ the commands

def inventory(root: Path) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        _inventory(root, discover(root, ShippedFamily(CLUSTER_RUNS, Path(scratch))))


def _inventory(root: Path, src: list[Source]) -> None:
    print(f"{'family':10s} {'p':>3s} {'pop':5s} {'mesh':11s} {'D':>5s} {'files':>5s} {'blocks':>6s} "
          f"{'KB':>6s}  {'written (first .. last)':35s} along  run")
    for s in src:
        m = s.settings
        mt = [local_time(f.stat().st_mtime) for f in s.files]
        along = (m.along or "-") + (f"@{m.search_along}" if m.search_along else "")
        print(f"{m.family:10s} {m.power if m.power else '-'!s:>3.3s} {m.population:5s} "
              f"{m.mesh or '-':11s} {m.budget if m.budget else '-'!s:>5.5s} {len(s.files):5d} "
              f"{len(set(s.block_ids)):6d} {sum(f.stat().st_size for f in s.files) / 1e3:6.0f}  "
              f"{stamp(min(mt))} .. {stamp(max(mt))}  {along}  {m.run}")
    rows = [s for s in src if s.settings.family != "collected"]
    print(f"{len(src)} runs; {sum(len(s.files) for s in rows)} row files, "
          f"{sum(f.stat().st_size for s in rows for f in s.files) / 1e6:.1f} MB "
          "(a collected file is counted once per name)")


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

    for line in check_copies(root, frames):
        print(" ", line)
    notes = collected_notes(frames, infos)
    for run, note in notes.items():
        info = infos[run]
        infos[run] = replace(info, note=f"{info.note}; {note}")

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
