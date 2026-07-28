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

import contextlib
import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO, cast

import segno
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict
from shapely.ops import unary_union

from reblock.contracts import Method, Screen, Source
from reblock.emit import region_map
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, block_depths
from reblock.render import google_maps_url
from scripts.compare_budgets import load_permeability_config, run_permeability_lenses
from scripts.gen_example_readme import write_readme

log = logging.getLogger(__name__)


def write_maps_qr(url: str, path: Path, *, scale: int = 4, border: int = 2) -> None:
    """Write a PNG QR code of `url` (e.g. the Google Maps locator) to `path`."""
    segno.make(url, error="m").save(str(path), scale=scale, border=border)


@contextlib.contextmanager
def _tee_to_file(path: Path) -> Iterator[None]:
    """Mirror stdout+stderr (and root logging at INFO) into `path` for the duration;
    restore after."""
    f = open(path, "w", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, *streams: TextIO) -> None: self._streams = streams
        def write(self, s: str) -> None:
            for st in self._streams:
                st.write(s)
        def flush(self) -> None:
            for st in self._streams:
                st.flush()

    orig_out, orig_err = sys.stdout, sys.stderr
    root = logging.getLogger()
    handler = logging.StreamHandler(f)
    handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    prev_level = root.level
    root.addHandler(handler)
    if prev_level == logging.NOTSET or prev_level > logging.INFO:
        root.setLevel(logging.INFO)
    try:
        sys.stdout = _Tee(orig_out, f)
        sys.stderr = _Tee(orig_err, f)
        yield
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        root.removeHandler(handler)
        root.setLevel(prev_level)
        f.flush()
        f.close()

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


def example_command(metric: str, city: str) -> str:
    base = f"pixi run python -m scripts.gen_multiblock_example {metric}"
    return base if city == "capetown" else f"{base} {city}"


# greedy_arterial_repulsion's default max_roads (15) is a fixed budget, so on large regions it
# undershoots: on density_compactness (~4677 parcels) 15 roads reach only ~4% displacement, well
# short of the D/P* lens thresholds, so its frontier stops mid-air. A per-region calibrated cap
# lets it reach past both cutoffs (density_compactness: k=40 -> ~19% disp / 68% perm, ~matching the
# other methods' extent). Regions not listed keep the default; the general
# region-adaptive fix (a displacement-target stop instead of a fixed count) is follow-up.
_ARTERIAL_MAX_ROADS = {"density_compactness": 40}


def main() -> None:
    metric_name = sys.argv[1]
    city = sys.argv[2] if len(sys.argv) > 2 else "capetown"
    # Cape Town examples live flat (examples/multiblock_<metric>); other cities nest under
    # examples/<city>/ so a metro's three variants stay grouped.
    out = (Path(f"examples/multiblock_{metric_name}") if city == "capetown"
           else Path(f"examples/{city}/multiblock_{metric_name}"))
    out.mkdir(parents=True, exist_ok=True)
    # Clear stale artifacts so a changed method set -- or the retired two-lens/external-internal
    # surface a committed example dir predates (lens_a_external.csv, curve_{external,internal}_
    # connectivity_*.png, depth_vs_road_*.png, displacement_*.png/*.csv,
    # frontier_{external,internal}_connectivity.csv) -- leaves no orphans (all regenerated below).
    for pattern in ("reblock_*.gif", "after_*.png", "curve_*.png", "depth_vs_road_*.png",
                    "displacement_*.png"):
        for stale in out.glob(pattern):
            stale.unlink()
    for name in ("displacement_table.csv", "displacement_vs_length.csv",
                "frontier_external_connectivity.csv", "frontier_internal_connectivity.csv",
                "lens_a_external.csv", "lens_b_matched.csv"):
        stale_path = out / name
        if stale_path.exists():
            stale_path.unlink()
    overrides = [
        f"metric={metric_name}", f"data={city}_full", "screen=dense_compact",
        "region_builder=dense_cluster", "region_builder.max_buildings=3000", "max_blocks=1",
        "all_methods.greedy_arterial_repulsion.candidate_policy=fixed",
        "+all_methods.greedy_arterial_repulsion.max_anchors=64",
        "all_methods.clearance_looped.base.depth_target=3",
        "all_methods.clearance_looped.base.max_roads=3000",
        "all_methods.clearance_looped.budget_frac=0.30",
        "all_methods.clearance_looped.search_radius_m=60",
        "all_methods.demand_looped.base.max_roads=3000",
        "all_methods.demand_looped.budget_frac=0.30",
        "all_methods.demand_looped.search_radius_m=60",
        "all_methods.euclidean_grid.spacing=250"]   # coarser grid -> budget in the pack
    if metric_name in _ARTERIAL_MAX_ROADS:
        overrides.append(
            f"all_methods.greedy_arterial_repulsion.max_roads={_ARTERIAL_MAX_ROADS[metric_name]}")
    with _tee_to_file(out / "run.log"):
        with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
            cfg = compose(config_name="compare_config", overrides=overrides)
        source = cast(Source, instantiate(cfg.data))
        screen = cast(Screen, instantiate(cfg.screen))
        region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))

        t0 = time.perf_counter()
        selection = screen.select(source) or []
        if not selection:
            raise SystemExit(f"metric={metric_name!r} flagged 0 blocks — check its gate/pre-filter")
        scores = cast("dict[str, float]", screen.selection_scores(source))   # type: ignore[attr-defined]
        total = len(source.block_geometries())
        log.info("screen: flagged %d/%d blocks (%.1fs)", len(selection), total,
                 time.perf_counter() - t0)

        source.block_ids = None                                              # type: ignore[attr-defined]
        t0 = time.perf_counter()
        region = build_regions(source, screen, region_builder, None, 1)[0]
        members = [b.block_id for b in region]
        region_parcels = sum(len(b.parcels) for b in region)
        log.info("region built: %d blocks / %d parcels (%.1fs)", len(members), region_parcels,
                 time.perf_counter() - t0)
        seed = selection[0]
        # build_regions narrowed source.block_ids to the members; clear it again (like run.py)
        # so the screen map spans the whole metro, not just the region neighbourhood.
        source.block_ids = None                                              # type: ignore[attr-defined]
        maps_url = google_maps_url(unary_union([b.boundary for b in region]), region[0].crs)
        write_maps_qr(maps_url, out / "maps_qr.png")

        region_map(source, [members], [[seed]], out,
                   selection=selection, depths=scores, metric_name=metric_name,
                   metric=getattr(screen, "metric", None))
        # region_map already writes screen.png/region.png (transparent, via save_render) at the
        # example naming -- no JPG flatten step needed.

        methods = {n: cast(Method, instantiate(cfg.all_methods[n]))
                   for n in ("greedy_arterial_repulsion", "clearance_looped", "euclidean_grid",
                             "demand_greedy_uniform", "demand_looped")}
        # osm_footpaths: the real as-built informal network, from a committed per-region OSM
        # snapshot (fetched once by scripts.fetch_desire_lines_snapshot) so the example
        # reproduces offline.
        snapshot = out / f"desire_lines_{seed}.geojson"
        if snapshot.exists():
            with open_dict(cfg):
                cfg.desire_source.snapshot = str(snapshot)
            methods["osm_footpaths"] = cast(Method, instantiate(cfg.all_methods.osm_footpaths))
            # Same snapshot: demand_greedy consumes desire lines as a routing PRIOR rather than
            # as its output, so it is only reproducible offline where osm_footpaths is.
            methods["demand_greedy"] = cast(Method, instantiate(cfg.all_methods.demand_greedy))
        params, matched_displacement, matched_permeability = load_permeability_config()
        run_permeability_lenses(region, methods, out, matched_displacement=matched_displacement,
                                matched_permeability=matched_permeability, params=params,
                                label=seed)

        depths = block_depths(source, members)
        dens = {b.block_id: len(b.parcels) / b.parcels.geometry.union_all().area * 1e4
                 for b in region}
        meta = {
            "metric": metric_name, "flagged": len(selection), "total_blocks": total,
            "deepest_block": seed, "deepest_depth": depths.get(seed, 0.0),
            "region_members": len(members), "region_parcels": region_parcels,
            "region_mean_depth": sum(depths.values()) / max(len(depths), 1),
            "region_mean_density_per_ha": sum(dens.values()) / max(len(dens), 1),
            "maps_url": maps_url,
            "command": example_command(metric_name, city),
            "maps_qr": "maps_qr.png",
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        write_readme(out, out, metric_name=metric_name, formula=_FORMULA[metric_name],
                     blurb=_BLURB[metric_name])
        print(f"wrote {out}: {len(members)} blocks / {meta['region_parcels']} parcels; "
              f"flagged {len(selection)}/{total}; README + maps + lens CSVs")


if __name__ == "__main__":
    main()
