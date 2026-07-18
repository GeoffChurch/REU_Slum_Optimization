"""One-off: fetch OSM desire-lines for a region/block and write a committed GeoJSON snapshot so the
examples reproduce osm_footpaths offline + byte-stable (no live Overpass call at example time).

Run (module form -- puts the repo root on sys.path so the data source's `from scripts...`
import resolves): `pixi run python -m scripts.fetch_desire_lines_snapshot <out.geojson>
<hydra override>...`

  # multiblock flagship (23-block region grown from 5810):
  ... examples/multiblock/desire_lines_5810.geojson \
      block_ids=[[ZAF.9.3.1_1_5810]] region_builder=dense_cluster region_builder.max_buildings=3000
  # method-comparison (single deep block 40972):
  ... examples/method-comparison/desire_lines_40972.geojson block_ids=[[ZAF.9.3.1_1_40972]]
"""
import sys
from pathlib import Path

import geopandas as gpd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict

from reblock.pipeline import build_regions


def main() -> None:
    out = Path(sys.argv[1])
    overrides = ["data=capetown_full", "max_blocks=1", *sys.argv[2:]]
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=overrides)
    source = instantiate(cfg.data)
    screen = instantiate(cfg.screen)
    region_builder = instantiate(cfg.region_builder)
    # Match the example orchestrator: with no explicit block_ids, screen-select and grow the top
    # region (groups=None), NOT an explicit seed -- growing from `[[seed]]` can yield a different
    # (larger) region than the screen's own top pick, giving a wrong, oversized bbox.
    groups = ([[str(b) for b in grp] for grp in cfg.block_ids]
              if cfg.get("block_ids") is not None else None)
    region = build_regions(source, screen, region_builder, groups, 1)[0]
    boundary = gpd.GeoSeries([b.boundary for b in region], crs=region[0].crs).union_all()
    bbox = gpd.GeoSeries([boundary], crs=region[0].crs).to_crs(4326).total_bounds
    # Fetch through the CONFIGURED desire_source (with snapshot forced off), so the endpoint is
    # overridable -- a big region can 504 the default Overpass; point it at a mirror via
    # `desire_source.endpoint=...`.
    with open_dict(cfg):
        cfg.desire_source.snapshot = None
    lines = instantiate(cfg.desire_source).desire_lines(
        (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), region[0].crs)
    print(f"fetched {len(lines)} desire-line ways -> {out}")
    assert len(lines) > 20, "sparse OSM coverage -- investigate before committing the snapshot"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines.to_crs(4326).to_file(out, driver="GeoJSON")


if __name__ == "__main__":
    main()
