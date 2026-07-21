"""Orchestrate one metric example variant end-to-end and emit its generated README.

A SINGLE metric drives everything -- the screen (proxy + fine), the region growth, and the map
colouring -- into `examples/multiblock_<metric>/`. Then the pure dir-reader generator
(`scripts.gen_example_readme`) turns the run's artifacts + a small `meta.json` into `README.md`, so
the prose can never drift from the numbers.

`osm_footpaths` is left out of the two-lens method comparison here: its desire-line snapshot is
region-specific and each metric picks a DIFFERENT region, so a fair osm run would need a fresh OSM
fetch per variant. `clearance` + `greedy_arterial` already demonstrate the two lenses on whatever
region the metric chose -- which is the point of the variant.

Run:  pixi run python -m scripts.gen_multiblock_example <depth|depth_density>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict
from PIL import Image
from shapely.ops import unary_union

from reblock.contracts import Method, Screen, Source
from reblock.emit import region_map
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, block_depths
from reblock.render import google_maps_url
from scripts.compare_budgets import run_two_lens
from scripts.gen_example_readme import write_readme

_FORMULA = {
    "depth": "depth = √(n·A)/P  →  true peel rings from a street",
    "depth_density": "depth × density  —  deep AND crowded",
    "density_compactness": "density × compactness = n/P²  —  dense, compact fabric (no peel)",
}
_BLURB = {
    "depth": "The deepest street-access fabric: how many parcels a home sits from a street, "
             "regardless of crowding.",
    "depth_density": "Deep and crowded at once — the metric that isolates the genuine informal "
                     "settlements and fades the deep-but-sparse blocks.",
    "density_compactness": "Dense and compact from geometry alone — the tightest, most built-up "
                           "blocks by building count per perimeter², found without ever peeling a "
                           "single parcel ring.",
}


def main() -> None:
    metric_name = sys.argv[1]
    city = sys.argv[2] if len(sys.argv) > 2 else "capetown"
    # Cape Town examples live flat (examples/multiblock_<metric>); other cities nest under
    # examples/<city>/ so a metro's three variants stay grouped.
    out = (Path(f"examples/multiblock_{metric_name}") if city == "capetown"
           else Path(f"examples/{city}/multiblock_{metric_name}"))
    out.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=[
            f"metric={metric_name}", f"data={city}_full", "screen=dense_compact",
            "region_builder=dense_cluster", "region_builder.max_buildings=3000", "max_blocks=1",
            "all_methods.clearance.max_roads=3000", "all_methods.clearance.depth_target=3",
            "all_methods.greedy_arterial_buildable.candidate_policy=fixed",
            "+all_methods.greedy_arterial_buildable.max_anchors=64",
            "all_methods.clearance_looped.base.depth_target=3",
            "all_methods.clearance_looped.base.max_roads=3000",
            "all_methods.clearance_looped.budget_frac=0.30",
            "all_methods.clearance_looped.search_radius_m=60"])
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))

    selection = screen.select(source) or []
    if not selection:
        raise SystemExit(f"metric={metric_name!r} flagged 0 blocks — check its gate/pre-filter")
    scores = cast("dict[str, float]", screen.selection_scores(source))   # type: ignore[attr-defined]
    total = len(source.block_geometries())

    source.block_ids = None                                              # type: ignore[attr-defined]
    region = build_regions(source, screen, region_builder, None, 1)[0]
    members = [b.block_id for b in region]
    seed = selection[0]
    # build_regions narrowed source.block_ids to the members; clear it again (like run.py) so the
    # screen map spans the whole metro, not just the region neighbourhood.
    source.block_ids = None                                              # type: ignore[attr-defined]
    maps_url = google_maps_url(unary_union([b.boundary for b in region]), region[0].crs)

    region_map(source, [members], [[seed]], out,
               selection=selection, depths=scores, metric_name=metric_name,
               metric=getattr(screen, "metric", None))
    for name in ("screen", "region"):
        png = out / f"{name}.png"
        if png.exists():
            Image.open(png).convert("RGB").save(out / f"{name}.jpg", quality=85)
            png.unlink()

    methods = {n: cast(Method, instantiate(cfg.all_methods[n]))
               for n in ("clearance", "greedy_arterial_buildable", "clearance_looped")}
    # osm_footpaths: the real as-built informal network, from a committed per-region OSM snapshot
    # (fetched once by scripts.fetch_desire_lines_snapshot) so the example reproduces offline.
    snapshot = out / f"desire_lines_{seed}.geojson"
    if snapshot.exists():
        with open_dict(cfg):
            cfg.desire_source.snapshot = str(snapshot)
        methods["osm_footpaths"] = cast(Method, instantiate(cfg.all_methods.osm_footpaths))
    # Continue clearance past its depth target on the depth curve ("to the longest road"): with
    # depth_target 0 it keeps clearing until the region is fully drained, so it reaches L_max. Only
    # clearance is extended -- it's the thorough clearer whose line is the point (it drains to ~1 at
    # the longest budget); arterial (sparse by design) and the fixed osm_footpaths stop at their own
    # converged lengths. Over-provisioning arterial's CELF doesn't scale to the big regions.
    with open_dict(cfg):
        cfg.all_methods.clearance.depth_target = 1     # clear to full drainage (bounded), not 0
        cfg.all_methods.clearance.max_roads = 10000
    extend = {"clearance": cast(Method, instantiate(cfg.all_methods.clearance))}
    run_two_lens(region, methods, 0.70, out, label=seed, extend=extend)

    depths = block_depths(source, members)
    dens = {b.block_id: len(b.parcels) / b.parcels.geometry.union_all().area * 1e4 for b in region}
    meta = {
        "metric": metric_name, "flagged": len(selection), "total_blocks": total,
        "deepest_block": seed, "deepest_depth": depths.get(seed, 0.0),
        "region_members": len(members), "region_parcels": sum(len(b.parcels) for b in region),
        "region_mean_depth": sum(depths.values()) / max(len(depths), 1),
        "region_mean_density_per_ha": sum(dens.values()) / max(len(dens), 1),
        "maps_url": maps_url,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    write_readme(out, out, metric_name=metric_name, formula=_FORMULA[metric_name],
                 blurb=_BLURB[metric_name])
    print(f"wrote {out}: {len(members)} blocks / {meta['region_parcels']} parcels; "
          f"flagged {len(selection)}/{total}; README + maps + lens CSVs")


if __name__ == "__main__":
    main()
