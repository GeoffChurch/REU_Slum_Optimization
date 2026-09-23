"""The research pool for a script that is not a Hydra app: `conf/donor_pool/` composed HERE, at the
script's own top level, and built by the typed loader.

`reblock.data.pools` composes nothing -- a pool is handed its built stages -- because composing
inside it breaks under an enclosing `@hydra.main` ("GlobalHydra is already initialized"). A plain
script has no enclosing Hydra, so this is the one place its pool is composed.
"""
from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from reblock.data.pools import ScreenedPool
from reblock.presets import load_research


def donor_pool(*overrides: str) -> ScreenedPool:
    """The `donor_pool` preset the overrides select -- `donor_pool=capetown`, or
    `donor_pool=shortlist_zone donor_pool.spec.stages.source.epsg=<epsg>`."""
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="config", overrides=list(overrides))
    return load_research(cfg.donor_pool, ScreenedPool)
