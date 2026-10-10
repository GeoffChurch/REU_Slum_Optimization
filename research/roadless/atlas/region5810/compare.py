"""Where the region's clearings fall, and 5810's per-block answer against the region's inside
5810."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

HERE = Path("/home/gchurchill/src/reblock/research/roadless")
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OUT = Path(sys.argv[1])
blocks = {b.block_id: b for b in
          common.build_blocks(["ZAF.9.3.1_1_5810@major", "ZAF.9.3.1_1_5810"])}
reg, blk = blocks["ZAF.9.3.1_1_5810@major"], blocks["ZAF.9.3.1_1_5810"]
rp, bp = np.asarray(reg.buildings.outlines), np.asarray(blk.buildings.outlines)
in5810 = shapely.contains(blk.boundary, shapely.centroid(rp))
area = shapely.area(rp)
# region index of each 5810 building, by identical outline
key = {p.wkb: i for i, p in enumerate(rp)}
to_reg = np.array([key.get(p.wkb, -1) for p in bp])
print("5810's buildings found in the region:", int((to_reg >= 0).sum()), "of", len(bp))
fc = pd.read_parquet(HERE / "fine_clearings.parquet")
per_block = {n: set(to_reg[np.asarray(c, dtype=int)]) for n, c in
             zip(fc[fc.block == blk.block_id].name, fc[fc.block == blk.block_id].cleared,
                 strict=True)
             if n in ("cheap", "default")}
print(f"region: {len(rp)} buildings, {in5810.sum()} in 5810 "
      f"({area[in5810].sum() / area.sum():.0%} of footprint area)")
for arm, per in (("cheap", "cheap"), ("simp", "default")):
    c = np.load(OUT / f"cleared_{arm}.npy")
    inn = in5810[c]
    print(f"{arm}: {len(c)} cleared, {inn.sum()} in 5810 "
          f"({area[c][inn].sum() / area[c].sum():.0%} of cleared area); "
          f"5810's per-block {per}: {len(per_block[per])}, "
          f"shared with the region's {len(per_block[per] & set(c))}")
