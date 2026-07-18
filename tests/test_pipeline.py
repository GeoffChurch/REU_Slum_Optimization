from typing import cast

from reblock.contracts import Screen, Source
from reblock.pipeline import RunOutput, _depth_fn


def test_runoutput_holds_selection_and_results() -> None:
    out = RunOutput(selection=["a", "b", "c"], results=[])
    assert out.selection == ["a", "b", "c"] and out.results == []


def test_depth_fn_prefers_screen_depths_over_peeling() -> None:
    # A flagged block's depth is a free dict lookup from the screen's precomputed selection_depths
    # (no parquet re-read); a non-flagged block falls to block_max_depth (0.0 for this non-Kblock
    # source). This keeps region growth fast -- the screen already peeled the flagged fabric.
    class _Src:
        blocks_path = "x"          # peel-capable marker

    class _Screen:
        def selection_depths(self, source: object) -> dict[str, float]:
            return {"flagged": 9.0}

    fn = _depth_fn(cast(Source, _Src()), cast(Screen, _Screen()))
    assert fn is not None
    assert fn("flagged") == 9.0    # screen dict lookup
    assert fn("other") == 0.0      # non-flagged -> block_max_depth -> 0.0 (not a KblockSource)


def test_depth_fn_none_for_non_peelable_source() -> None:
    # No blocks_path -> the builder uses its proxy fallback (depth_fn is None).
    class _Bare:
        pass

    class _Screen:
        pass

    assert _depth_fn(cast(Source, _Bare()), cast(Screen, _Screen())) is None
