"""Footprint AREA cleared at Lens A (10% of homes displaced): greedy (whole buildings) vs the road
arms (corridor overlap), as a share of the block's total footprint area."""
import sys
from pathlib import Path

sys.path.insert(0, 'research/roadless')
import numpy as np
import pandas as pd
import shapely

import common

HERE = Path('research/roadless')
from reblock.budget import prefix_to_displacement, road_corridor
from reblock.derivations import propose

blocks = {b.block_id: b for b in common.build_blocks(common.recipients())}
arms = common.arms()
done = sorted(p.stem for p in (HERE / "clear_rows_M4_h0.5").glob("*.parquet"))
rng = np.random.default_rng(2)
rows = []
for bid in sorted(rng.choice(done, 40, replace=False)):
    b = blocks[bid]
    polys = np.asarray(b.buildings.outlines)
    area = shapely.area(polys)
    g = pd.read_parquet(HERE / "clear_rows_M4_h0.5" / f"{bid}.parquet").sort_values("step")
    stop = g[g.D >= 0.10 - 1e-9].step.iloc[0]
    rem = g[(g.step >= 1) & (g.step <= stop)].cleared.to_numpy()
    rows.append(dict(block=bid, arm="clear_bldg", area_share=area[rem].sum() / area.sum(),
                     mean_area_ratio=area[rem].mean() / area.mean()))
    for arm in ("cycle_native", "resistance_lp", "greedy_arterial_access_displacement"):
        pre = prefix_to_displacement(b, propose(arms[arm], b).roads, 0.10)
        c = road_corridor(pre)
        taken = shapely.area(shapely.intersection(polys, c))
        rows.append(dict(block=bid, arm=arm, area_share=taken.sum() / area.sum(),
                         mean_area_ratio=np.nan))
d = pd.DataFrame(rows)
print(d.groupby("arm")[["area_share", "mean_area_ratio"]].median().round(3))
