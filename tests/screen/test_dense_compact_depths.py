from pathlib import Path

from reblock.data.kblock import KblockSource
from reblock.screen.dense_compact import DenseCompactScreen

_ROOT = Path(__file__).resolve().parent.parent


def _src() -> KblockSource:
    return KblockSource(_ROOT / "data/kblock/blocks_dji_sample.parquet",
                        _ROOT / "data/kblock/buildings_dji_sample.parquet", "dji")


def test_select_returns_ids_and_selection_depths_maps_them() -> None:
    # select() still returns plain block_ids (protocol unchanged); selection_depths returns the same
    # ids mapped to their true max access-depth, and the two agree on membership.
    screen = DenseCompactScreen(min_buildings=1)
    ids = screen.select(_src())
    depths = screen.selection_depths(_src())
    assert ids is not None
    assert all(isinstance(b, str) for b in ids)
    assert set(depths) == set(ids)                       # same blocks
    assert all(d >= 1.0 for d in depths.values())        # every flagged block is >= 1 ring deep


def test_screen_selection_returns_pairs_deepest_first() -> None:
    from reblock.derivations import ScreenSelectionInput, screen_selection
    from reblock.derive_graph import source_hash
    src = _src()
    inp = ScreenSelectionInput(
        source_hash=source_hash(src.blocks_path, src.buildings_path),
        blocks_path=str(src.blocks_path), buildings_path=str(src.buildings_path),
        depth_proxy_min=1.5, mean_depth_min=1.3, max_depth_min=None, k_min=None, min_buildings=1)
    pairs = screen_selection(inp)
    assert pairs and all(isinstance(b, str) and isinstance(d, float) for b, d in pairs)
    depths = [d for _, d in pairs]
    assert depths == sorted(depths, reverse=True)        # deepest-first (the ranking order)
