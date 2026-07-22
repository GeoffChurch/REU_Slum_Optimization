"""Regenerate the single-block method-comparison flagship example, fully reproducibly.

Reblocks ONE deep block (`ZAF.9.3.1_1_40972`) with every method -- including `topology`, which only
runs on a single block -- then, from ONE `propose` per method, emits the metric-basis curves +
frontier CSVs (external/internal connectivity + displacement, via `emit.compare_report`) AND a
before-heatmap plus one after-heatmap per method (`render_before`/`render_after`). Self-logs to
`run.log`, so the captured log is the source of truth for the hand-written README's figures.

Unlike `reblock.compare` (curves only), this renders the per-method before/after visuals too, so the
whole example -- curves AND images -- reproduces from one command with no hand-placed assets.

Run:  pixi run python -m scripts.gen_method_comparison
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from shapely.ops import unary_union

from reblock.budget import (
    access_benefit,
    building_radii,
    commute_ratio_benefit,
    cost_benefit_curve,
    displacement_curve,
)
from reblock.compare import MethodCurve
from reblock.contracts import Method, Screen, Source
from reblock.derivations import propose
from reblock.emit import _displaced_points, compare_report, pct_displaced, pct_paved
from reblock.eval.kcomplexity import KComplexityEval
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder
from reblock.render import frame_bbox, google_maps_url, render_after, render_before, save_render
from scripts.gen_multiblock_example import _tee_to_file

# The deepest block in a topology-tractable size window (see the README); auto-picked once, pinned
# here so the flagship reproduces exactly. `greedy_arterial_repulsion` is capped to 8 roads (sparse
# by design); `osm_footpaths` loads the committed OSM snapshot beside this file.
BLOCK_ID = "ZAF.9.3.1_1_40972"
METHODS = ("topology", "clearance", "greedy_arterial_repulsion", "clearance_looped",
           "euclidean_grid", "osm_footpaths")
OUT = Path("examples/method-comparison")
CORRIDOR_M = 3.0

log = logging.getLogger("gen_method_comparison")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Clear stale renders so a changed method set leaves no orphans (all regenerated below); leaves
    # the committed OSM snapshot + README + this run's CSVs (rewritten in place) untouched.
    for stale in (*OUT.glob("after_*.jpg"), *OUT.glob("before.jpg"),
                  *OUT.glob("curve_*.png"), *OUT.glob("displacement*.png")):
        stale.unlink()
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
        block = region[0]
        maps_url = google_maps_url(unary_union([b.boundary for b in region]), block.crs)
        log.info("%s map: %s", block.block_id, maps_url)

        radii = building_radii(block.building_points, CORRIDOR_M)
        block_area = float(block.parcels.geometry.union_all().area)
        frame = frame_bbox(block.parcels)
        # One propose per method -- reused for BOTH the curves and the renders (topology is the slow
        # pole; proposing it twice would double the runtime).
        proposals = {n: propose(m, block) for n, m in methods.items()}

        curves: list[MethodCurve] = []
        for name, prop in proposals.items():
            assert prop.roads is not None and not prop.roads.empty, f"{name}: no roads proposed"
            roads = prop.roads   # narrowed to a non-empty GeoDataFrame by the assert
            pp = pct_paved(roads, CORRIDOR_M, block_area)
            pd_ = pct_displaced(roads, CORRIDOR_M, block.building_points, radii)
            ext = cost_benefit_curve(block, roads, benefit_fn=access_benefit)
            internal = cost_benefit_curve(block, roads, benefit_fn=commute_ratio_benefit)
            disp = displacement_curve(block, roads, radii, corridor_m=CORRIDOR_M)
            for metric, curve in (("external_connectivity", ext),
                                  ("internal_connectivity", internal), ("displacement", disp)):
                curves.append(MethodCurve(name, block.block_id, metric, curve, pp, pd_))
        # method_order = the full registry, so a method reads the same hue here as in the
        # multiblock examples (compare_report's colour contract).
        compare_report(curves, OUT, method_order=[str(k) for k in cfg.all_methods])

        # Before + per-method after heatmaps on ONE shared depth scale. access_before is the
        # status-quo (method-independent), so any proposal's score fixes vmax.
        kc_eval = KComplexityEval()
        kc0 = kc_eval.score(block, proposals[METHODS[0]])
        vmax = int(kc0.fields["access_before"].max())
        save_render(render_before(block, kc0.fields["access_before"], vmax=vmax, frame=frame),
                    OUT / "before.jpg")
        plt.close("all")
        for name, prop in proposals.items():
            kc = kc_eval.score(block, prop)
            fig = render_after(block, prop, kc.fields["access_after"], vmax=vmax, metrics=kc,
                               frame=frame, displaced_points=_displaced_points(block, prop))
            save_render(fig, OUT / f"after_{name}.jpg")
            plt.close(fig)

        # Terminal frontier point per method per axis -- the README's figure source.
        for metric in ("external_connectivity", "internal_connectivity", "displacement"):
            for mc in sorted((c for c in curves if c.metric == metric),
                             key=lambda c: -c.curve.benefit[-1]):
                if metric == "displacement":
                    log.info("%s %s: %.1f displaced (%.1f%% of homes)", block.block_id, mc.method,
                             mc.curve.benefit[-1], mc.pct_displaced * 100)
                else:
                    log.info("%s %s %s: benefit=%.3f at %.0f m (%.1f%% paved)", metric,
                             block.block_id, mc.method, mc.curve.benefit[-1], mc.curve.cost[-1],
                             mc.pct_paved * 100)
        print(f"wrote {OUT}: {len(METHODS)} methods; curves + before/after renders + run.log")


if __name__ == "__main__":
    main()
