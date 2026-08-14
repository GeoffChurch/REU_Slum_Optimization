"""The Permeability page's figure set: the egress graph, drawn four ways on one block.

Two layers (edge width from conductance, then from current) x two states (no roads, then a real
method's roads). The conductance pair teaches the clearance-fraction mesh and what a road does to
it; the current pair teaches drainage, and is the image where adding a road visibly concentrates
flow into the new corridor.

WHY THIS IS NOT PART OF gen_example. It shares that pipeline's block and roads exactly -- same
pinned block, same config, same content-addressed derivation cache -- but iterating on a FIGURE's
design must not cost a ten-method comparison run. This loads one block and one method and takes
seconds.

Run:  pixi run python -m scripts.gen_perm_graph
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from reblock.budget import prefix_to_permeability
from reblock.compare import load_permeability_config
from reblock.contracts import Method, Screen, Source
from reblock.derivations import propose
from reblock.perm_graph import GRAPH_LAYERS, permeability_graph
from reblock.permeability import permeability
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder
from reblock.render import frame_bbox, render_graph, save_render

log = logging.getLogger(__name__)

VARIANT = "method_comparison"      # pins ZAF.9.3.1_1_40972; see conf/example/method_comparison.yaml
METHOD = "clearance"
OUT = Path("examples/perm-graph")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config",
                      overrides=[f"+example={VARIANT}", "data=capetown_full"])
    pcfg = load_permeability_config()
    params = pcfg.params

    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    groups = [list(g) for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, int(cfg.max_blocks))[0]
    assert len(region) == 1, "this figure set is single-block by design"
    block = region[0]

    method = cast(Method, instantiate(cfg.all_methods[METHOD]))
    roads = cast(GeoDataFrame, propose(method, block).roads)
    prefix, reached = prefix_to_permeability(block, roads, pcfg.matched_permeability, params)
    if not reached:
        raise SystemExit(
            f"{METHOD} never reached P*={pcfg.matched_permeability} on {block.block_id}; the "
            f"'after' figure would not be the Lens-B prefix the site publishes")
    log.info("block %s: %d parcels, %s prefix %.0f m", block.block_id, len(block.parcels),
             METHOD, float(prefix.geometry.length.sum()))

    before = permeability_graph(block, None, params)
    after = permeability_graph(block, prefix, params)

    # Shared scales. Both figures are derived FIRST so every image can be put on one scale per
    # quantity -- the same discipline compare_budgets applies to vmax and frame. Without it the
    # before/after pair is two pictures at two zoom levels of ink, and teaches nothing.
    frame = frame_bbox(block.parcels)
    vmax = max(float(before.potential.max()), float(after.potential.max()))
    # Mesh-only norm (fix round 1, Finding A): a road-raised edge's conductance/current sits ~2-3
    # orders of magnitude above the surrounding mesh (a real, physically-correct trunk, not an
    # outlier to rob p99 of its robustness). Pooling p99 over ALL edges lets those ~65 upgraded
    # edges set the norm, which crushes every mesh edge's width into a band under 1% of
    # [_EDGE_LW_MIN, _EDGE_LW_MAX] -- invisible. Excluding upgraded edges from the norm (both
    # states; `before.upgraded` is all-False so this is a no-op there) puts the mesh's own spread
    # back on a legible scale; upgraded edges then clip at the maximum, which loses no information
    # because colour (`_ROAD_COLOR`) already marks them -- see render_graph's docstring.
    norms = {layer: max(_p99(read(before)[~before.upgraded]), _p99(read(after)[~after.upgraded]))
             for layer, read in GRAPH_LAYERS.items()}

    for state, fig_data, prefix_roads in (("before", before, None), ("after", after, prefix)):
        for layer in GRAPH_LAYERS:
            fig = render_graph(fig_data, block, layer=layer, vmax=vmax,
                               width_norm=norms[layer], frame=frame, roads=prefix_roads)
            path = OUT / f"graph_{layer}_{state}.png"
            save_render(fig, path)
            plt.close(fig)
            log.info("wrote %s", path)

    meta = {
        "block_id": block.block_id,
        "method": METHOD,
        "p_star": pcfg.matched_permeability,
        "permeability_before": 0.0,      # by definition: 1 - P(no roads)/P(no roads)
        "permeability_after": permeability(block, prefix, params),
        "road_m": float(prefix.geometry.length.sum()),
        "n_parcels": int(before.n),
        "n_edges": int(len(before.rows)),
        "n_upgraded": int(after.upgraded.sum()),
    }
    (OUT / "perm_graph.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", OUT / "perm_graph.json")


def _p99(x: np.ndarray) -> float:
    """A robust maximum for edge-width normalization: one trunk edge orders of magnitude above the
    rest would otherwise flatten the whole mesh to the hairline floor."""
    return float(np.percentile(np.abs(x), 99)) if len(x) else 0.0


if __name__ == "__main__":
    main()
