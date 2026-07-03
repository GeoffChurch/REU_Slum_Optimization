from pathlib import Path

from reblock.contracts import Block
from reblock.data.shapefile import ShapefileSource

PHULE = (Path(__file__).resolve().parents[2] / "ext" / "topology" / "examples" / "data"
         / "phule_nagar_v6.shp")


def test_source_yields_metric_blocks() -> None:
    region = ShapefileSource(PHULE, region_id="phule").region()
    blocks = list(region.blocks)
    assert len(blocks) >= 1
    b = blocks[0]
    assert isinstance(b, Block) and b.crs.is_projected
    assert not b.parcels.empty and "parcel_id" in b.parcels.columns
    assert b.boundary.area > 0 and b.block_id.startswith("phule_")
