"""One-off: render one after-heatmap per method for a region, every method's roads truncated to a
MATCHED added-road-length budget (the sparsest method's total) so the comparison is fair (Task 7's
multiblock example grid).

Run (module form -- mirrors scripts/fetch_desire_lines_snapshot.py's Hydra bootstrapping):
  pixi run python -m scripts.render_methods_matched <out_dir> <m1,m2,...> <hydra override>...

  e.g. examples/multiblock clearance,greedy_arterial_buildable,osm_footpaths \
       data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
       block_ids=[[ZAF.9.3.1_1_5810]] all_methods.clearance.max_roads=3000 \
       all_methods.greedy_arterial_buildable.candidate_policy=fixed \
       +all_methods.greedy_arterial_buildable.max_anchors=64 \
       desire_source.snapshot=examples/multiblock/desire_lines_osm_5810.geojson

Each method is reblocked once (`region_reblock`), then its roads are truncated
(`truncate_to_length`) to the matched budget and access-depth is RE-SCORED on the truncated roads
via `KComplexityEval` -- the same eval call `region_reblock`/`pipeline.run` make (`ev.score(block,
proposal)`), just invoked directly on a Proposal wrapping the truncated roads instead of the
method's full output. The truncated Proposal's `block_identity` is cleared (`None`) so
`Proposal.identity` is `None` and the `derive()` memoization layer (reblock.derive_graph) bypasses
its cache for this call -- reusing the method's own `proposal_id`/`block_identity` verbatim would
give the truncated roads the SAME cache key as the method's already-computed full proposal, and a
warm cache (in-process or the on-disk L2) would silently hand back the wrong (untruncated)
access-depth. Everything else on the Proposal (roads aside) is preserved -- `params["corridor_m"]`
is what `_displaced_points` and `render_after`'s corridor draw both key on.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from reblock.budget import matched_budget, truncate_to_length
from reblock.contracts import Method, Screen, Source
from reblock.emit import _displaced_points
from reblock.eval.kcomplexity import KComplexityEval
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, region_reblock
from reblock.render import frame_bbox, render_after, save_render


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    method_names = sys.argv[2].split(",")
    overrides = ["max_blocks=1", *sys.argv[3:]]
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=overrides)
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    groups = [[str(b) for b in g] for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, 1)[0]
    methods = {n: cast(Method, instantiate(cfg.all_methods[n])) for n in method_names}

    # 1) reblock each method jointly over the region (evals=[]: this script scores its own,
    # budget-truncated access-depth below, not the method's full-roads metrics).
    results = {n: region_reblock(region, m, []) for n, m in methods.items()}
    lengths = {n: float(cast(GeoDataFrame, r.proposal.roads).geometry.length.sum())
               for n, r in results.items()}
    budget = matched_budget(lengths)

    # 2) per method: truncate to the matched budget, re-score access-depth on the truncated roads,
    # render. vmax is fixed from the FIRST method's access_before (method-independent -- same
    # region block content every time -- so every after-render shares one colour scale, mirroring
    # emit._render_block_group).
    kc_eval = KComplexityEval()
    vmax: int | None = None
    for n, r in results.items():
        block = r.block
        roads_t = truncate_to_length(block, cast(GeoDataFrame, r.proposal.roads), budget)
        truncated = replace(r.proposal, roads=roads_t, block_identity=None)
        kc = kc_eval.score(block, truncated)
        if vmax is None:
            vmax = int(kc.fields["access_before"].max())
        fig = render_after(
            block, truncated, kc.fields["access_after"], vmax=vmax, metrics=kc,
            frame=frame_bbox(block.parcels), displaced_points=_displaced_points(block, truncated),
        )
        save_render(fig, out_dir / f"after_{n}.jpg")
        plt.close(fig)
    print(f"rendered {len(results)} methods at matched budget {budget:.0f} m -> {out_dir}")


if __name__ == "__main__":
    main()
