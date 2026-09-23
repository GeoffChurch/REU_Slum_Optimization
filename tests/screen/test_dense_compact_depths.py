from pathlib import Path

from reblock.data.kblock import KblockSource
from reblock.metric import AbsoluteGate, Compactness, Density, Depth, PercentileGate, Product
from reblock.screen.dense_compact import DenseCompactScreen

_ROOT = Path(__file__).resolve().parent.parent


def _src() -> KblockSource:
    return KblockSource(_ROOT / "data/kblock/blocks_dji_sample.parquet",
                        _ROOT / "data/kblock/buildings_dji_sample.parquet", "dji")


def test_depth_metric_selects_and_scores_by_true_depth() -> None:
    # metric=Depth with a permissive absolute gate: select() returns ids, selection_scores maps them
    # to the fine score (true max peel depth for Depth), and they agree on membership.
    screen = DenseCompactScreen(Depth(), AbsoluteGate(1.0), proxy_keep_n=1000,
                                min_buildings=1)
    ids = screen.select(_src())
    scores = screen.selection_scores(_src())
    assert ids and set(scores) == set(ids)
    assert all(s >= 1.0 for s in scores.values())      # gate floor


def test_a_survivor_the_fine_pass_cannot_build_is_never_flagged() -> None:
    """At `min_buildings=1` some of this sample's survivors are too small to tessellate, so the
    fine pass returns no depth for them. A stand-in depth of 0 used to take its place -- and a
    100th-percentile gate keeps every score it is given, so blocks that can never be reblocked were
    flagged."""
    import reblock.screen.dense_compact as dc
    seen: dict[str, set[str]] = {}
    real = dc._survivor_depths

    def spy(survivors: list[str], blocks_path: str, buildings_path: str,
            min_buildings: int) -> list[tuple[str, float]]:
        rows = real(survivors, blocks_path, buildings_path, min_buildings)
        seen["unbuilt"] = set(survivors) - {b for b, _ in rows}
        return rows

    dc._survivor_depths = spy
    try:
        screen = DenseCompactScreen(Depth(), PercentileGate(100.0), min_buildings=1)
        ranked = dc._compute_selection(screen._selection_input(_src()))
    finally:
        dc._survivor_depths = real
    assert seen["unbuilt"], "precondition: the sample has survivors too small to build"
    assert not seen["unbuilt"] & {b for b, _ in ranked}


def test_density_compactness_metric_skips_the_peel() -> None:
    # needs_peel=False -> the fine pass must NOT call _survivor_depths (no Voronoi/peel).
    import reblock.screen.dense_compact as dc
    calls = {"n": 0}
    real = dc._survivor_depths

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    dc._survivor_depths = spy
    try:
        screen = DenseCompactScreen(Product([Density(), Compactness()]),
                                    PercentileGate(20.0), min_buildings=1)
        ids = screen.select(_src())
    finally:
        dc._survivor_depths = real
    assert ids                       # still selects
    assert calls["n"] == 0           # peel skipped entirely
