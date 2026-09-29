"""The 220 consensus recipients and their blocks (the two helpers of the lost research module, standalone)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path("/home/gchurchill/src/reblock")


def recipients() -> list[str]:
    rows = pd.read_parquet(REPO / "data/benchmarks/consensus_matrix.parquet", columns=["recipient"])
    return sorted(rows["recipient"].unique())


def build_blocks(ids: list[str]) -> list:
    from hydra import compose, initialize_config_dir

    from reblock.presets import load_stages
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "conf")):
        cfg = compose(config_name="compare_config",
                      overrides=["data=capetown_full", "buildings=footprints"])
    source = load_stages(cfg).source
    blocks = sorted(source.restricted(ids).region().blocks, key=lambda b: b.block_id)
    assert all(type(b.buildings).__name__ == "Footprints" for b in blocks)
    return blocks
