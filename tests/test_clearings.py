"""The committed stored clearings (research/roadless/clearings.py, clearings_export.py): the format
reads back what it wrote, and a read refuses a block that is not the one its positions were
committed for.

The module is research code (no reblock closure reaches it, the gate does not lint or type-check
research/), loaded here by path. Its tests build their own store under tmp_path: a block of six
squares, the store's files written by the writer under test, read by the reader under test.

Fault-injected where a docstring says so: the fault was applied, the test seen to fail, the code
restored.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

ROADLESS = Path(__file__).resolve().parents[1] / "research" / "roadless"


def _load(name: str, register_as: str | None = None) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROADLESS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[register_as or name] = mod
    spec.loader.exec_module(mod)
    return mod


# The exporter imports the library as `clearings` and puts research/roadless on sys.path; load the
# library under that name first so both are one module, and put sys.path back.
_saved_path = list(sys.path)
clearings: Any = _load("clearings")
exporter: Any = _load("clearings_export")
sys.path[:] = _saved_path


@dataclass
class Toy:
    """A block as a read asks of one: its id and its buildings' outlines in row order."""
    block_id: str
    outlines: gpd.GeoSeries

    @property
    def buildings(self) -> Toy:
        return self


SQUARES = [shapely.box(3 * i, 0, 3 * i + 2, 2 + i) for i in range(6)]    # six different areas


def toy(squares: list[Any] | None = None, block_id: str = "toy") -> Toy:
    return Toy(block_id, gpd.GeoSeries(SQUARES if squares is None else squares))


def info(run: str, mode: str, **kw: Any) -> Any:
    fields: dict[str, Any] = dict(
        run=run, family="greedy", mode=mode, spec="S0.01cat", power=2.0, demand="w",
        population="area", along="uni", search_along="", mesh="h0.5", budget=None,
        source="clear_rows_toy", source_files=1, source_bytes=1000,
        written_first="2026-10-13 00:00", written_last="2026-10-13 00:01", metric="preB",
        blocks=1, rows=2, ids=3, note="")
    fields.update(kw)
    return clearings.RunInfo(**fields)


def nested_rows(block: str = "toy") -> pd.DataFrame:
    return pd.DataFrame({
        "block": [block, block, block], "step": [1, 2, 3],
        "D": [0.1, 0.30000000000000004, 1 / 3], "perm": [0.5, 1e-06, np.nan],
        "cleared": [np.array([3, 1]), np.array([5]), np.array([0, 4])]})


@pytest.fixture
def store(tmp_path: Path) -> Any:
    s = clearings.Store(tmp_path / "clearings")
    s.write_prints([clearings.BlockPrint.of(toy())])
    s.write_table("greedy", nested_rows())
    s.write_table("single", nested_rows().iloc[1:2].assign(step=1))
    s.write_runs([info("greedy", "added"), info("single", "whole", rows=1, ids=1)])
    return s


# ------------------------------------------------------------------ the format round-trips

def test_nested_rows_read_back_as_the_cumulative_clearing(store: Any) -> None:
    """Mode `added`: the clearing through a step is every row's positions up to it, in the order
    they were cleared (not sorted)."""
    b = toy()
    assert store.cleared("greedy", b, step=1).tolist() == [3, 1]
    assert store.cleared("greedy", b, step=2).tolist() == [3, 1, 5]
    assert store.cleared("greedy", b, step=3).tolist() == [3, 1, 5, 0, 4]


def test_whole_rows_read_back_as_themselves(store: Any) -> None:
    """Mode `whole`: a row is the clearing, not an increment."""
    assert store.cleared("single", toy(), step=1).tolist() == [5]


def test_a_whole_run_does_not_accumulate(tmp_path: Path) -> None:
    s = clearings.Store(tmp_path)
    s.write_prints([clearings.BlockPrint.of(toy())])
    s.write_table("levels", nested_rows())
    s.write_runs([info("levels", "whole")])
    assert s.cleared("levels", toy(), step=3).tolist() == [0, 4]
    s.write_runs([info("levels", "added")])
    assert s.cleared("levels", toy(), step=3).tolist() == [3, 1, 5, 0, 4]


def test_floats_read_back_exactly(store: Any) -> None:
    t = store.table("greedy")
    assert t["D"].tolist() == [0.1, 0.30000000000000004, 1 / 3]
    assert t["perm"].iloc[1] == 1e-06
    assert np.isnan(t["perm"].iloc[2])                 # blank in the file, NaN in the frame
    assert t["step"].tolist() == [1, 2, 3]
    assert [c.tolist() for c in t["cleared"]] == [[3, 1], [5], [0, 4]]


def test_a_file_holds_the_required_columns_and_those_with_a_value(store: Any) -> None:
    header = (store.root / "greedy.csv").read_text().splitlines()[0]
    assert header == "block,step,D,cleared,perm"        # no enclosed, free_*, t, ... column
    rows = nested_rows().assign(enclosed=[0.0, np.nan, np.nan],
                                free_D=[0.0, np.nan, np.nan],
                                free_cleared=[np.array([], dtype=np.int64), None, None])
    store.write_table("free", rows)
    assert (store.root / "free.csv").read_text().splitlines()[0] == (
        "block,step,D,cleared,perm,enclosed,free_D")     # free_cleared: blank all through, dropped
    t = store.table("free")
    assert t["enclosed"].iloc[0] == 0.0 and np.isnan(t["enclosed"].iloc[1])
    assert t["free_D"].iloc[0] == 0.0                   # present and zero: free first held nothing


def test_the_files_are_deterministic_and_sorted(tmp_path: Path) -> None:
    a, b = clearings.Store(tmp_path / "a"), clearings.Store(tmp_path / "b")
    rows = nested_rows()
    a.write_table("r", rows)
    b.write_table("r", rows.iloc[::-1].reset_index(drop=True))     # shuffled input, same file
    assert (a.root / "r.csv").read_bytes() == (b.root / "r.csv").read_bytes()


def test_run_info_round_trips_blank_as_none(store: Any) -> None:
    got = store.runs()["greedy"]
    assert got == info("greedy", "added")
    assert got.budget is None and got.search_along == ""
    note = 'has, a comma and "quotes"'
    store.write_runs([info("a", "whole", budget=0.05, power=None, note=note)])
    assert store.runs()["a"].budget == 0.05 and store.runs()["a"].power is None
    assert store.runs()["a"].note == note


def test_the_columns_are_one_closed_set() -> None:
    """COLUMNS has no duplicate, REQUIRED is inside it, and the module's own documentation names
    every column (a column added without its meaning fails here)."""
    names = [c.name for c in clearings.COLUMNS]
    assert len(names) == len(set(names))
    assert set(clearings.REQUIRED) <= set(names)
    doc = clearings.__doc__
    for n in names:
        assert re.search(rf"^\s{{4}}{re.escape(n)}\b|\b{re.escape(n)}\b[,\s]", doc, re.M), n


def test_a_column_outside_the_set_is_refused_both_ways(store: Any) -> None:
    with pytest.raises(ValueError, match="outside COLUMNS"):
        store.write_table("bad", nested_rows().assign(surprise=1.0))
    (store.root / "bad.csv").write_text("block,step,D,cleared,surprise\ntoy,1,0.1,1,2\n")
    with pytest.raises(KeyError, match="surprise"):
        store.table("bad")


def test_steps_must_run_from_one_without_gaps(store: Any) -> None:
    with pytest.raises(ValueError, match="not 1, 2"):
        store.write_table("gap", nested_rows().assign(step=[1, 2, 4]))
    (store.root / "gap.csv").write_text("block,step,D,cleared\ntoy,1,0.1,1\ntoy,3,0.2,2\n")
    store.write_runs([info("gap", "added"), *store.runs().values()])
    with pytest.raises(ValueError, match="not 1, 2"):
        store.cleared("gap", toy(), step=1)


def test_a_missing_run_block_or_step_is_a_key_error(store: Any) -> None:
    with pytest.raises(KeyError, match="no run 'nope'"):
        store.cleared("nope", toy(), step=1)
    with pytest.raises(KeyError, match="no step 9"):
        store.cleared("greedy", toy(), step=9)
    other = toy(block_id="other")
    store.write_prints([*store.prints().values(), clearings.BlockPrint.of(other)])
    with pytest.raises(KeyError, match="holds no clearing of block 'other'"):
        store.cleared("greedy", other, step=1)


def test_positions_beyond_the_block_are_refused(store: Any) -> None:
    (store.root / "far.csv").write_text("block,step,D,cleared\ntoy,1,0.1,2 99\n")
    store.write_runs([info("far", "whole"), *store.runs().values()])
    with pytest.raises(ValueError, match="outside 0..5"):
        store.cleared("far", toy(), step=1)


# ------------------------------------------------------------------ the fingerprint guard

def test_the_fingerprint_is_pure_and_sees_every_kind_of_change() -> None:
    base = clearings.fingerprint(SQUARES)
    assert clearings.fingerprint(list(SQUARES)) == base
    assert clearings.fingerprint(gpd.GeoSeries(SQUARES)) == base
    moved = [*SQUARES[:2], shapely.box(6, 0, 8.000001, 4), *SQUARES[3:]]
    swapped = [SQUARES[1], SQUARES[0], *SQUARES[2:]]
    assert clearings.fingerprint(moved) != base                    # a vertex
    assert clearings.fingerprint(swapped) != base                  # the same footprints, reordered
    assert clearings.fingerprint(SQUARES[:-1]) != base             # a building dropped
    assert clearings.fingerprint([*SQUARES, SQUARES[0]]) != base   # a building added
    assert len({clearings.fingerprint(x) for x in (SQUARES, moved, swapped)}) == 3


def test_a_reordered_block_is_refused(store: Any) -> None:
    """The guard's job: the same six footprints in another row order (a reader or a source that
    changed) would hand the committed positions to other buildings, and the count would not
    notice. Fault-injected: with the fingerprint comparison removed from Store.check this test and
    the moved-vertex case below failed (nothing raised); with the count comparison removed the
    count test failed; with the fingerprint made blind to the row order this test and the
    fingerprint's own test failed."""
    swapped = toy([SQUARES[1], SQUARES[0], *SQUARES[2:]])
    with pytest.raises(clearings.StaleClearing, match="reordered"):
        store.cleared("greedy", swapped, step=1)


def test_only_the_fingerprint_catches_a_reorder(
        store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The negative control of the test above: with the fingerprint made to agree with the stored
    one, the reordered block is read without a murmur, its positions now naming the other
    buildings. So the refusal above is the fingerprint's, not the count's."""
    swapped = toy([SQUARES[1], SQUARES[0], *SQUARES[2:]])
    stored = store.prints()["toy"].fingerprint
    monkeypatch.setattr(clearings, "fingerprint", lambda _outlines: stored)
    assert store.cleared("greedy", swapped, step=1).tolist() == [3, 1]


@pytest.mark.parametrize("squares", [
    pytest.param([*SQUARES[:2], shapely.box(6, 0, 8.000001, 4), *SQUARES[3:]], id="moved-vertex"),
    pytest.param([*SQUARES, shapely.box(30, 0, 32, 2)], id="a-building-added"),
    pytest.param(SQUARES[:-1], id="a-building-dropped"),
])
def test_a_changed_block_is_refused(store: Any, squares: list[Any]) -> None:
    with pytest.raises(clearings.StaleClearing):
        store.cleared("greedy", toy(squares), step=1)


def test_a_block_of_another_count_is_refused_by_its_count(store: Any) -> None:
    with pytest.raises(clearings.StaleClearing, match="6 buildings, the block in hand has 5"):
        store.cleared("greedy", toy(SQUARES[:-1]), step=1)


def test_an_uncommitted_block_is_a_key_error(store: Any) -> None:
    with pytest.raises(KeyError, match="no clearing of block 'unknown'"):
        store.cleared("greedy", toy(block_id="unknown"), step=1)


def test_a_missing_footprint_cannot_be_pinned() -> None:
    with pytest.raises(ValueError, match="missing"):
        clearings.fingerprint([SQUARES[0], None])


# ------------------------------------------------------------------ the exporter's checks

def block_data(squares: list[Any] | None = None) -> Any:
    b = toy(squares)
    return exporter.BlockData(print=clearings.BlockPrint.of(b),
                              areas=np.asarray(shapely.area(np.asarray(b.outlines))))


def check(mode: str, population: str, rs: list[dict[str, Any]], data: Any) -> None:
    exporter.check_rows("r", mode, population, rs, data)


def rows(*steps: tuple[list[int], float | None], n: int | None = 6) -> list[dict[str, Any]]:
    out = []
    for k, (cleared, d) in enumerate(steps, 1):
        r: dict[str, Any] = dict(block="toy", step=k, cleared=np.array(cleared))
        if d is not None:
            r["D"] = d
        if n is not None:
            r["n"] = n
        out.append(r)
    return out


def test_the_exporter_recomputes_d_from_the_positions() -> None:
    data = {"toy": block_data()}
    share = float(shapely.area(SQUARES[3]) / sum(shapely.area(s) for s in SQUARES))
    good = rows(([3], share))
    check("added", "area", good, data)
    # the source stored no D (a collected set): it is filled in, as the area share
    bare = rows(([3], None))
    check("added", "area", bare, data)
    assert bare[0]["D"] == pytest.approx(share, abs=1e-15)
    # a D that other buildings would make is refused: what a reordered block does to a row
    with pytest.raises(ValueError, match="share is"):
        check("added", "area", rows(([0], share)), data)


def test_the_exporter_counts_buildings_by_the_population_the_run_names() -> None:
    data = {"toy": block_data()}
    check("added", "count", rows(([3, 1], 2 / 6)), data)
    with pytest.raises(ValueError, match="share is"):
        check("added", "count", rows(([3, 1], 0.1)), data)


def test_the_exporter_refuses_a_block_of_another_size_a_range_overrun_and_a_repeat() -> None:
    data = {"toy": block_data()}
    with pytest.raises(ValueError, match="the row says 7 buildings"):
        check("added", "area", rows(([3], None), n=7), data)
    with pytest.raises(ValueError, match="outside 0..5"):
        check("added", "area", rows(([9], None)), data)
    with pytest.raises(ValueError, match="twice"):
        check("added", "area", rows(([3], None), ([3, 4], None)), data)
    check("whole", "area", rows(([3], None), ([3, 4], None)), data)   # not nested


def test_the_audit_refuses_a_column_it_does_not_know() -> None:
    cols = ["simp_D", "simp_perm", "simp_cleared", "block", "n", "power", "plan", "along", "t"]
    exporter.audit(cols, exporter.SIMP, Path("f"))
    with pytest.raises(ValueError, match="'brand_new'"):
        exporter.audit([*cols, "brand_new"], exporter.SIMP, Path("f"))


def test_a_runs_row_is_written_once_unless_its_replacement_is_named(tmp_path: Path) -> None:
    """The optimizers' no-overwrite guard (clearings.write_once): a second write of one identity
    raises and leaves the first row; `replace` names the replacement. Fault-injected: with
    `_exclusive` an os.replace (the check removed) the second write replaced the first and this
    failed; clearings_check.py `overwrite` runs it through the optimizers."""
    f = tmp_path / "relax_rows" / "m" / "plan" / "D0.05" / "b_p1.parquet"
    clearings.write_once(pd.DataFrame({"perm": [0.1]}), f, replace=False)
    with pytest.raises(clearings.Exists):
        clearings.check_new(f, replace=False)
    with pytest.raises(clearings.Exists):
        clearings.write_once(pd.DataFrame({"perm": [0.2]}), f, replace=False)
    assert pd.read_parquet(f)["perm"].tolist() == [0.1]
    assert not [g for g in f.parent.iterdir() if g != f]          # no temp file left behind
    clearings.check_new(f, replace=True)
    clearings.write_once(pd.DataFrame({"perm": [0.3]}), f, replace=True)
    assert pd.read_parquet(f)["perm"].tolist() == [0.3]
    assert clearings.replace_flag(["one", "--replace", "x"]) == (["one", "x"], True)


def test_merging_keeps_what_the_sources_lost_adds_what_is_new_and_flags_an_overwrite(
        tmp_path: Path) -> None:
    s = clearings.Store(tmp_path)
    s.write_table("r", pd.concat([nested_rows("a"), nested_rows("b")], ignore_index=True))
    new = pd.concat([nested_rows("b"), nested_rows("c")], ignore_index=True)
    m = exporter.merge(s, "r", new)
    assert (m.added, m.kept, m.changed) == (1, 1, [])             # c is new, a only the store has
    assert sorted(m.frame["block"].unique()) == ["a", "b", "c"]
    over = new.copy()
    over.loc[(over["block"] == "b") & (over["step"] == 2), "cleared"] = pd.Series(
        [np.array([4])], index=over.index[(over["block"] == "b") & (over["step"] == 2)])
    assert exporter.merge(s, "r", over).changed == ["b"]          # the rewritten row is a conflict


# ------------------------------------------------------------------ the sources: this code's metric

RUNTIME = exporter.metrics.RUNTIME         # this code's complex metric (its digest this code's)
PLAN = "fw0.q3.i10.e0.0001.k1s1e-06"
RUN_DIR = f"relax_rows/{RUNTIME}/{PLAN}/D0.05"


@pytest.mark.parametrize("name, picker, power, along", [
    (f"clear_rows_SW3-1.5-1.2xS0.01catr0.005m8.P64w8D0.05_p1_{RUNTIME}",
     "SW3-1.5-1.2xS0.01catr0.005m8.P64w8D0.05", "1", RUNTIME),
    (f"clear_rows_SIMP{PLAN}_p2_{RUNTIME}@e0.05", f"SIMP{PLAN}", "2", f"{RUNTIME}@e0.05"),
])
def test_a_greedy_directory_name_parses_into_its_settings(
        name: str, picker: str, power: str, along: str) -> None:
    """clear.rows_dir's names (the greedy's format under a complex metric) parse back."""
    m = exporter.GREEDY_DIR.fullmatch(name)
    assert m is not None
    assert (m["picker"], m["power"], m["along"]) == (picker, power, along)
    assert exporter.this_code(m["along"])


@pytest.mark.parametrize("name", [
    "clear_rows_M4_h0.5", "clear_rows_S0.01cat_h1_area_p2",
    "clear_rows_S0.01cat_h0.5_area_p2_ss100k2n2r30@ss100k0.5n2r30",
    "clear_rows_SW3-1.5-1.2xS0.01catr0.005m8.P64w8D0.05_h0.5_area_p2",
])
def test_the_raster_greedy_directories_are_records(name: str) -> None:
    """The retired raster's greedy-format directories (`_h<h>` in the name) are records: no
    pattern of this code's reads them, so no source does. Fault-injected: with `_record` answering
    False for every path, every case failed."""
    assert exporter._record(name + "/x.parquet")


def test_only_this_codes_metric_is_a_source(tmp_path: Path) -> None:
    """Under relax_rows, this code's complex metric's runs are read; the raster's metric names and
    another code's complex (another digest) are records, listed and not read. Fault-injected: with
    `this_code` answering True for every name, the raster's directories were read and this
    failed."""
    root = tmp_path / "roadless"
    for metric in (RUNTIME, "uni", "uni+apwfs", "cx+shy0.5+s0.125r1b1+R0.5g4x2q4",
                   RUNTIME.replace(exporter.gap_clear.CODE, "00000000")):
        d = root / "relax_rows" / metric / PLAN / "D0.05"
        d.mkdir(parents=True)
        simp_row("blk", [3], along=metric, enclosed=0.0).to_parquet(d / "blk_p1.parquet")
    [src] = exporter.SimpFamily(store="relax_rows").sources(root)
    assert src.settings.run == f"{RUN_DIR}.p1" and src.settings.along == RUNTIME
    assert sorted(exporter.records(root)) == sorted(
        f"relax_rows/{m}" for m in ("uni", "uni+apwfs", "cx+shy0.5+s0.125r1b1+R0.5g4x2q4",
                                    RUNTIME.replace(exporter.gap_clear.CODE, "00000000")))


# ------------------------------------------------------------------ shipped rows, since overwritten


def share(cleared: list[int]) -> float:
    areas = np.asarray(shapely.area(np.asarray(SQUARES)))
    return float(areas[cleared].sum() / areas.sum())


def simp_row(block: str, cleared: list[int], *, along: str = RUNTIME,
             **extra: Any) -> pd.DataFrame:
    """A relax.py `one` row, as relax_rows holds it."""
    return pd.DataFrame([dict(block=block, n=6, power=2.0, plan=PLAN, along=along,
                              simp_perm=0.25, simp_D=share(cleared), simp_grey=0.01,
                              simp_cleared=cleared, simp_x=[0.5] * 6, t=12.0, **extra)])


def put_tar(path: Path, members: dict[str, bytes], mtime: int = 1_000_000_000) -> None:
    import io
    import tarfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            info_ = tarfile.TarInfo(name)
            info_.size, info_.mtime = len(data), mtime
            tf.addfile(info_, io.BytesIO(data))


def parquet_bytes(df: pd.DataFrame) -> bytes:
    import io
    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()


def test_a_row_a_cluster_run_was_shipped_and_a_later_run_overwrote_is_recovered(
        tmp_path: Path) -> None:
    """The road-b run shipped the 60 large blocks' stored clearings at 2026-10-11 16:23; the
    free-first runs of 20:17 overwrote 12 blocks' rows in place. The shipped copies are the only
    ones left of those rows: the family takes a shipped row that differs from the local one, takes
    it once however many runs shipped it, takes nothing that equals the local row, and names the
    run after the earliest that shipped it. Fault-injected: with the comparison against the local
    file removed every shipped row was taken, and the identical ones failed this test."""
    local = tmp_path / "roadless"
    member = f"research/roadless/{RUN_DIR}/blk_p2.parquet"
    same_member = f"research/roadless/{RUN_DIR}/same_p2.parquet"
    old, new = simp_row("blk", [1, 2], enclosed=0.0), simp_row("blk", [3], enclosed=0.0)
    (local / RUN_DIR).mkdir(parents=True)
    new.to_parquet(local / RUN_DIR / "blk_p2.parquet")                  # what the local file says
    same = simp_row("same", [4], enclosed=0.0)
    same.to_parquet(local / RUN_DIR / "same_p2.parquet")
    shipped = {member: parquet_bytes(old), same_member: parquet_bytes(same)}
    runs = tmp_path / "runs"
    put_tar(runs / "road-b" / "remote" / "inputs" / "blk.tar", shipped)
    put_tar(runs / "road-b-r1" / "remote" / "inputs" / "blk.tar", shipped)        # shipped again
    put_tar(runs / "later" / "remote" / "inputs" / "blk.tar",
            {member: parquet_bytes(new)}, mtime=2_000_000_000)                    # shipped as now
    fam = exporter.ShippedFamily(runs, tmp_path / "scratch")
    [src] = fam.sources(local)
    assert src.settings.run == f"{RUN_DIR}.p2@road-b"
    assert src.block_ids == ("blk",) and len(src.files) == 1
    [row] = src.read(src.files[0])
    assert row["cleared"].tolist() == [1, 2]                      # the old row, not the local one
    mtime = src.files[0].stat().st_mtime
    assert mtime == 1_000_000_000                                 # the row's own writing time


def test_a_row_must_state_what_it_leaves_walled_in(tmp_path: Path) -> None:
    """Every row of this code's metric states `enclosed` (B (i), free first's), and its run is the
    B scorer's; a row without it raises. (The retired raster's rows before lifted.Placement had
    none, and were told apart by their date: records now, read by no code.) Fault-injected: with
    the check removed, the row without `enclosed` was read and this failed."""
    data = {"blk": block_data()}
    rows = tmp_path / "roadless"
    (rows / RUN_DIR).mkdir(parents=True)

    def source_of(file: Path) -> Any:
        [src] = exporter.SimpFamily(store="relax_rows").sources(rows)
        assert src.files == (file,)
        return src

    f = rows / RUN_DIR / "blk_p2.parquet"
    simp_row("blk", [3], enclosed=0.0, free_D=0.0, free_cleared=[]).to_parquet(f)
    _frame, got = exporter.read_run(source_of(f), data)
    assert got.metric == "B"
    simp_row("blk", [3]).to_parquet(f)
    with pytest.raises(ValueError, match="B \\(i\\) unstated"):
        exporter.read_run(source_of(f), data)


# ------------------------------------------------------------------ the committed store itself

def test_the_committed_store_is_whole_and_says_what_it_holds() -> None:
    """research/roadless/clearings/ as committed: runs.csv lists exactly the run files there are,
    every file reads under the closed column set with each block's steps 1, 2, ..., every block
    has a print, every position lies inside its block, and each run's line says what its file
    holds (blocks, rows, positions, and the scorer: `B` where every row states `enclosed`, `preB`
    where none does). Fault-injected: a position past its block's last building, a row dropped
    from a file, a file dropped from the index and an `enclosed` added to a preB run each failed
    this test."""
    store = clearings.Store.committed()
    runs, prints = store.runs(), store.prints()
    on_disk = {p.relative_to(store.root).with_suffix("").as_posix()
               for p in store.root.rglob("*.csv")} - {"blocks", "runs"}
    assert on_disk == set(runs)
    assert all(len(p.fingerprint) == 64 and p.n > 0 for p in prints.values())
    for run, info in runs.items():
        t = store.table(run)
        assert info.mode in ("added", "whole")
        assert t["block"].nunique() == info.blocks, run
        assert len(t) == info.rows, run
        assert sum(len(c) for c in t["cleared"]) == info.ids, run
        for block, g in t.groupby("block"):
            assert block in prints, (run, block)
            assert g["step"].tolist() == list(range(1, len(g) + 1)), (run, block)
            n = prints[str(block)].n
            for c in g["cleared"]:
                assert len(c) == 0 or (c.min() >= 0 and c.max() < n), (run, block)
        stated = (t["enclosed"].notna() if "enclosed" in t.columns
                  else pd.Series(False, index=t.index))
        scorer = "B" if stated.all() else "preB" if not stated.any() else "mixed"
        assert info.metric == scorer, (run, info.metric, scorer)
