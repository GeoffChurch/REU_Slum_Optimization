"""The SINGLE entry point for every committed example. One script, one pipeline.

A variant is a config file (`conf/example/<name>.yaml`) and nothing else. It sets exactly two things
that matter -- WHICH REGION to grade and WHICH METHODS to run -- plus the README prose. There is no
second script and no per-example Python branch beyond the one this difference forces:

    block_ids: null      grow a region with the screen + region_builder (the multiblock variants)
    block_ids: [[...]]   pin exact blocks instead (method-comparison, which needs a single block
                         because `topology`, the prior art, crashes on a multi-block region)

Everything downstream is identical: the same lens run, the same renders, the same `meta.json`, and
the same dir-reading README generator (`scripts.gen_example_readme`), so no example's prose can
drift from its numbers. A pinned variant simply omits the screen/region-growth keys from meta and
the generator skips those sections.

`osm_footpaths` joins any variant that has a committed OSM snapshot beside it -- the real as-built
network, a reference rather than a competitor, and the one entry that can fail to reach P*.

`flow_paths` is deliberately absent: it exists as the real-footpath MIMICRY probe (it reproduces
footpath position but not geometry), a question about realism rather than reblocking quality. It
stays registered in `all_methods` and is worth re-adding if the question becomes "what do the
synthetic methods miss" -- on the depth region it reached P* with FEWER metres than any other
method, so it is not dominated.

Run:  pixi run python -m scripts.gen_example <variant> [city]
"""
from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import segno
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from shapely.ops import unary_union

from reblock.compare import load_permeability_config
from reblock.contracts import ScoringScreen
from reblock.emit import region_map
from reblock.pipeline import build_regions
from reblock.presets import load_method, load_methods, load_stages
from reblock.region import block_depths
from reblock.render import google_maps_url
from scripts.compare_budgets import run_permeability_lenses
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

def example_command(variant: str, city: str) -> str:
    base = f"pixi run python -m scripts.gen_example {variant}"
    return base if city == "capetown" else f"{base} {city}"


def main() -> None:
    variant = sys.argv[1]
    city = sys.argv[2] if len(sys.argv) > 2 else "capetown"
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config",
                      overrides=[f"+example={variant}", f"data={city}_full"])
    metric_name = str(cfg.metric_name) if "metric_name" in cfg else variant
    # Cape Town examples live flat; other cities nest under examples/<city>/ so a metro's
    # variants stay grouped.
    slug = str(cfg.example.slug)
    out = Path(f"examples/{slug}") if city == "capetown" else Path(f"examples/{city}/{slug}")
    out.mkdir(parents=True, exist_ok=True)
    # Clear stale artifacts so a changed method set -- or the retired two-lens/external-internal
    # surface a committed example dir predates (lens_a_external.csv, curve_{external,internal}_
    # connectivity_*.png, depth_vs_road_*.png, displacement_*.png/*.csv,
    # frontier_{external,internal}_connectivity.csv) -- leaves no orphans (all regenerated below).
    # `*.jpg` is in the list because renders used to be JPG-flattened and are not any more (see
    # region_map below): 122 unreferenced fossils survived every regeneration because the old
    # cleanup globbed only .png/.gif, and one of them was still linked from the top-level README.
    for pattern in ("reblock_*.gif", "after_*.png", "curve_*.png", "depth_vs_road_*.png",
                    "displacement_*.png", "*.jpg"):
        for stale in out.glob(pattern):
            stale.unlink()
    for name in ("displacement_table.csv", "displacement_vs_length.csv",
                "frontier_external_connectivity.csv", "frontier_internal_connectivity.csv",
                "lens_a_external.csv", "lens_b_matched.csv"):
        stale_path = out / name
        if stale_path.exists():
            stale_path.unlink()
    pinned = cfg.block_ids                # None -> grow a region; a list -> pin these
    with _tee_to_file(out / "run.log"):
        stages = load_stages(cfg)
        source, screen, region_builder = stages.source, stages.screen, stages.region_builder
        # Every registered method, before the screen spends its minutes: a broken entry fails now.
        registry = load_methods(cfg.all_methods)

        # The ONE branch the two region modes force. A pinned variant has no screen selection to
        # report, so it also omits the screen/region keys from meta and the README generator (which
        # guards every one of them) simply skips those sections.
        selection: list[str] = []
        scores: dict[str, float] = {}
        total = 0
        # Growing a region needs a screen that scores blocks: the selection seeds it, and
        # `region_map` colours by the same metric and grades on the counter the screen resolved.
        scoring: ScoringScreen | None = None
        if pinned is None:
            if not isinstance(screen, ScoringScreen):
                raise SystemExit(f"{slug}: growing a region needs a scoring screen, but "
                                 f"{type(screen).__name__} does not score blocks -- pin "
                                 f"block_ids, or use screen=dense_compact")
            scoring = screen
            t0 = time.perf_counter()
            selection = scoring.select(source) or []
            if not selection:
                raise SystemExit(
                    f"metric={metric_name!r} flagged 0 blocks — check its gate/pre-filter")
            scores = scoring.selection_scores(source)
            total = len(source.block_geometries())
            log.info("screen: flagged %d/%d blocks (%.1fs)", len(selection), total,
                     time.perf_counter() - t0)
            source.block_ids = None                                          # type: ignore[attr-defined]

        t0 = time.perf_counter()
        groups = None if pinned is None else [list(g) for g in pinned]
        # `seed_rank` picks WHICH of the screen's ranked blocks seeds this example, so two
        # variants can share one screen instead of needing one screen each. build_regions returns
        # one region per seed group in rank order, so it must be asked for at least rank+1 of them.
        seed_rank = int(cfg.get("seed_rank", 0))
        region = build_regions(source, screen, region_builder, groups,
                               max(int(cfg.max_blocks), seed_rank + 1))[seed_rank]
        members = [b.block_id for b in region]
        region_parcels = sum(len(b.parcels) for b in region)
        log.info("region built: %d blocks / %d parcels (%.1fs)", len(members), region_parcels,
                 time.perf_counter() - t0)
        seed = selection[seed_rank] if len(selection) > seed_rank else members[0]
        # build_regions narrowed source.block_ids to the members; clear it again (like run.py)
        # so the screen map spans the whole metro, not just the region neighbourhood.
        source.block_ids = None                                              # type: ignore[attr-defined]
        maps_url = google_maps_url(unary_union([b.boundary for b in region]), region[0].crs)
        write_maps_qr(maps_url, out / "maps_qr.png")

        if scoring is not None:
            region_map(source, [members], [[seed]], out,
                       selection=selection, depths=scores, metric_name=metric_name,
                       metric=scoring.metric,
                       # The SAME counter the screen ranked on, so the map grades blocks on the
                       # number the pipeline used rather than the source's vendor column.
                       counts=scoring.counts)
            # region_map already writes screen.png/region.png (transparent, via save_render) at
            # the example naming -- no JPG flatten step needed.

        methods = {str(n): registry[n] for n in cfg.methods}
        # osm_footpaths: the real as-built informal network, from a committed per-region OSM
        # snapshot (fetched once by scripts.fetch_desire_lines_snapshot) so the example
        # reproduces offline.
        snapshot = out / f"desire_lines_{seed}.geojson"
        if snapshot.exists():
            with open_dict(cfg):
                cfg.desire_source.snapshot = str(snapshot)
            methods["osm_footpaths"] = load_method(cfg.all_methods.osm_footpaths)
        run_permeability_lenses(region, methods, out, pcfg=load_permeability_config(),
                                label=seed)

        depths = block_depths(source, members)
        dens = {b.block_id: len(b.parcels) / b.parcels.geometry.union_all().area * 1e4
                 for b in region}
        meta: dict[str, object] = {
            "metric": metric_name,
            "deepest_block": seed, "deepest_depth": depths.get(seed, 0.0),
            "region_parcels": region_parcels,
            # The ids, not just the count: `scripts/preflight_regen.py` diffs this to say whether
            # a regeneration would re-run the methods (hours) or only re-render (minutes), and a
            # count alone cannot distinguish a changed region from a same-size different one.
            "region_member_ids": [str(b) for b in members],
            "region_mean_depth": sum(depths.values()) / max(len(depths), 1),
            "region_mean_density_per_ha": sum(dens.values()) / max(len(dens), 1),
            "maps_url": maps_url,
            "command": example_command(variant, city),
            "maps_qr": "maps_qr.png",
        }
        if pinned is None:
            # screen/region-growth keys only exist for a grown region; the README generator guards
            # each of them, so a pinned variant just renders without those sections.
            meta |= {"flagged": len(selection), "total_blocks": total,
                     "region_members": len(members)}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        write_readme(out, out, metric_name=metric_name, formula=str(cfg.example.formula),
                     blurb=str(cfg.example.blurb))
        scope = (f"flagged {len(selection)}/{total}" if pinned is None
                 else f"pinned {len(members)} block(s)")
        print(f"wrote {out}: {len(members)} blocks / {region_parcels} parcels; "
              f"{scope}; {len(methods)} methods; README + maps + lens CSVs")


if __name__ == "__main__":
    main()
