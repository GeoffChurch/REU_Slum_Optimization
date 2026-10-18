"""Access depth of every parcel under each lineup method's whole network (does the arterial
saturate its objective, i.e. stop because every parcel already fronts a road?)."""
import sys, time, glob
sys.path.insert(0, "research/roadless")
import numpy as np, pandas as pd, geopandas as gpd
import common
from reblock.derive.access import STREET_TOL, ParcelAdjacency, parcel_access_layers, past_every_parcel

ROWS = "research/roadless/road_rows/lineup-caf14fa2/scratch/networks"
ids = sys.argv[1].split(",")
out = []
for b in common.build_blocks(ids):
    t = time.perf_counter()
    adj = ParcelAdjacency.of(b, STREET_TOL)
    d0 = parcel_access_layers(adj, None, unreached=past_every_parcel)
    for name in ["greedy_arterial_access_displacement", "cycle_native", "resistance_lp", "clearance"]:
        try:
            roads = gpd.read_parquet(f"{ROWS}/{b.block_id}/{name}.parquet")
        except FileNotFoundError:
            continue
        d = parcel_access_layers(adj, roads, unreached=past_every_parcel)
        out.append(dict(block=b.block_id, method=name, parcels=len(d), d0_max=float(d0.max()),
                        d0_burden=float((d0**2).sum()), max_depth=float(d.max()),
                        frac_depth1=float((d <= 1).mean()), burden=float((d**2).sum()),
                        n_seg=len(roads), road_km_per_km2=roads.length.sum() / b.boundary.area * 1000))
    print(b.block_id, f"{time.perf_counter()-t:.1f}s", flush=True)
df = pd.DataFrame(out)
df.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 250)
print(df.to_string())
