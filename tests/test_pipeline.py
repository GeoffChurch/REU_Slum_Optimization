from pathlib import Path

from reblock.data.kblock import KblockSource
from reblock.pipeline import RunOutput, sample, select_blocks
from reblock.screen.identity import IdentityScreen

_KB = Path(__file__).resolve().parent / "data" / "kblock"


def test_select_blocks_yields_in_selection_priority_order() -> None:
    # select_blocks returns built blocks in the screen's priority order (worst-first for
    # DenseCompactScreen), NOT the alphabetical/parquet order region() builds in.
    src = KblockSource(str(_KB / "blocks_capetown_sample.parquet"),
                       str(_KB / "buildings_capetown_sample.parquet"), region_id="ct")
    order = ["ZAF.9.3.1_1_44882", "ZAF.9.3.1_1_21719"]   # reverse-alphabetical on purpose
    _, blocks = select_blocks(src, IdentityScreen(order), max_blocks=10)
    assert [b.block_id for b in blocks] == order


def test_sample_takes_first_n_of_a_list() -> None:
    assert sample(["a", "b", "c", "d"], 2) == ["a", "b"]


def test_sample_passes_through_all() -> None:
    assert sample(None, 5) is None            # None = ALL


def test_sample_n_larger_than_selection() -> None:
    assert sample(["a", "b"], 10) == ["a", "b"]


def test_runoutput_holds_selection_and_results() -> None:
    out = RunOutput(selection=["a", "b", "c"], results=[])
    assert out.selection == ["a", "b", "c"] and out.results == []
