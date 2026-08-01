"""Regenerate the single-block method-comparison flagship example, fully reproducibly.

Reblocks ONE deep block (`ZAF.9.3.1_1_40972`) with every method -- including `topology`, which only
runs on a single block -- then hands the region + methods to `run_permeability_lenses` (the same
driver `gen_multiblock_example.py` uses): ONE propose per method, the permeability-vs-displacement
frontier, both lenses' outcome tables + before/after renders (both heatmap colorings), and the
per-method reblock GIF. Self-logs to `run.log`, so the captured log is the source of truth for the
hand-written README's figures.

Unlike `reblock.compare` (curves only), this renders the per-method before/after visuals too, so the
whole example -- curves AND images -- reproduces from one command with no hand-placed assets.

Run:  pixi run python -m scripts.gen_method_comparison
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from shapely.ops import unary_union

from reblock.contracts import Method, Screen, Source
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder
from reblock.render import google_maps_url
from scripts.compare_budgets import load_permeability_config, run_permeability_lenses
from scripts.gen_multiblock_example import _tee_to_file

# The deepest block in a topology-tractable size window (see the README); auto-picked once, pinned
# here so the flagship reproduces exactly. `greedy_arterial_repulsion` is capped to 8 roads (sparse
# by design); `osm_footpaths` loads the committed OSM snapshot beside this file.
BLOCK_ID = "ZAF.9.3.1_1_40972"
METHODS = ("topology", "clearance", "clearance_looped", "cycle_native",
           "euclidean_grid", "osm_footpaths")
OUT = Path("examples/method-comparison")

log = logging.getLogger("gen_method_comparison")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Clear stale renders -- including the retired three-curve/single-before-image surface the
    # committed dir predates (before.jpg with no coloring suffix, curve_{external,internal}_
    # connectivity_*.png, displacement_*.png/*.csv, frontier_{external,internal}_connectivity.csv)
    # -- so a changed method set leaves no orphans (all regenerated below); leaves the committed
    # OSM snapshot + README untouched.
    for pattern in ("after_*.png", "before_*.png", "frontier_*.png", "reblock_*.gif", "curve_*.png",
                    "displacement_*.png"):
        for stale in OUT.glob(pattern):
            stale.unlink()
    for name in ("before.jpg", "displacement_table.csv", "displacement_vs_length.csv",
                "frontier_external_connectivity.csv", "frontier_internal_connectivity.csv"):
        stale_path = OUT / name
        if stale_path.exists():
            stale_path.unlink()
    with _tee_to_file(OUT / "run.log"):
        with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
            cfg = compose(config_name="compare_config", overrides=[
                "data=capetown_full", f"block_ids=[[{BLOCK_ID}]]", "max_blocks=1",
                "all_methods.greedy_arterial_repulsion.max_roads=8",
                f"desire_source.snapshot={OUT}/desire_lines_40972.geojson"])
        source = cast(Source, instantiate(cfg.data))
        screen = cast(Screen, instantiate(cfg.screen))
        region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
        methods = {n: cast(Method, instantiate(cfg.all_methods[n])) for n in METHODS}

        region = build_regions(source, screen, region_builder, [[BLOCK_ID]], 1)[0]
        assert len(region) == 1, "method-comparison is single-block (topology needs it)"
        maps_url = google_maps_url(unary_union([b.boundary for b in region]), region[0].crs)
        log.info("%s map: %s", region[0].block_id, maps_url)

        params, matched_displacement, matched_permeability = load_permeability_config()
        rows = run_permeability_lenses(
            region, methods, OUT, matched_displacement=matched_displacement,
            matched_permeability=matched_permeability, params=params, label=BLOCK_ID)

        for r in rows:
            log.info("%s [lens A D=%.2f] permeability=%.3f at %.0f m", r.method,
                     matched_displacement, r.disp_permeability, r.disp_road_m)
            mark = f"reached P*={matched_permeability:.2f}" if r.reached else "UNREACHED"
            log.info("%s [lens B P*=%.2f] %s, %.1f%% displaced at %.0f m", r.method,
                     matched_permeability, mark, r.perm_displacement * 100, r.perm_road_m)
        print(f"wrote {OUT}: {len(METHODS)} methods; frontier + both lenses + renders + run.log")


if __name__ == "__main__":
    main()
