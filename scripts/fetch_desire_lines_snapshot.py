"""One-off: fetch desire-lines for a region/block from a chosen source and write a committed GeoJSON
snapshot, so the examples reproduce dream_come_true offline + byte-stable (no live Overpass/imagery
call at example time). The source is whichever `desire_source` group is selected -- OSM (Overpass)
or imagery (Esri World Imagery + wide-corridor detector) -- so one script serves both variants.

Run (module form -- puts the repo root on sys.path so the data source's `from scripts...` import
resolves): `pixi run python -m scripts.fetch_desire_lines_snapshot <out.geojson> <overrides>...`

  # OSM (dream_come_true_osm) -- multiblock flagship (23-block region grown from 5810):
  ... examples/multiblock/desire_lines_osm_5810.geojson desire_source=osm \
      block_ids=[[ZAF.9.3.1_1_5810]] region_builder=dense_cluster region_builder.max_buildings=3000
  # imagery (dream_come_true_cv) -- same region, wide-corridor detection:
  ... examples/multiblock/desire_lines_cv_5810.geojson desire_source=imagery \
      block_ids=[[ZAF.9.3.1_1_5810]] region_builder=dense_cluster region_builder.max_buildings=3000
  # method-comparison (single deep block 40972), either source:
  ... examples/method-comparison/desire_lines_cv_40972.geojson desire_source=imagery \
      block_ids=[[ZAF.9.3.1_1_40972]]
"""
import sys
from pathlib import Path

import geopandas as gpd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from reblock.pipeline import build_regions


def main() -> None:
    out = Path(sys.argv[1])
    overrides = ["data=capetown_full", "max_blocks=1", *sys.argv[2:]]
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=overrides)
    source = instantiate(cfg.desire_source)
    screen = instantiate(cfg.screen)
    region_builder = instantiate(cfg.region_builder)
    data = instantiate(cfg.data)
    groups = [[str(b) for b in grp] for grp in cfg.block_ids]
    region = build_regions(data, screen, region_builder, groups, 1)[0]
    boundary = gpd.GeoSeries([b.boundary for b in region], crs=region[0].crs).union_all()
    bbox = gpd.GeoSeries([boundary], crs=region[0].crs).to_crs(4326).total_bounds
    lines = source.desire_lines(
        (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), region[0].crs)
    total_m = float(lines.geometry.length.sum())
    print(f"{type(source).__name__}: {len(lines)} desire-lines, {total_m:.0f} m total -> {out}")
    assert len(lines) > 0, "no desire-lines detected -- investigate before committing the snapshot"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines.to_crs(4326).to_file(out, driver="GeoJSON")


if __name__ == "__main__":
    main()
