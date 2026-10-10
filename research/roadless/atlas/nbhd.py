"""5810's neighbourhood: blocks touching it (and their neighbours), sizes, OSM road classes."""
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyogrio

out = sys.argv[1]
b = gpd.read_parquet("/home/gchurchill/.cache/reblock/blocks_capetown_full.parquet",
                     columns=["block_id", "building_count", "geometry"])
b["block_id"] = b.block_id.astype(str)
utm = b.estimate_utm_crs()
b = b.to_crs(utm)
c = b[b.block_id == "ZAF.9.3.1_1_5810"].iloc[0]
g = c.geometry
print("5810:", c.building_count, "bldgs", f"{g.area/1e3:.0f}k m2", "holes", len(g.interiors),
      f"isoperimetric {4*3.14159*g.area/g.length**2:.2f}", "bounds", [round(x) for x in g.bounds])
win = g.buffer(600)
near = b[b.intersects(win)].copy()
near["area_k"] = near.area / 1e3
near["touch"] = near.distance(g) < 0.5
touching = near[near.touch].sort_values("building_count", ascending=False)
print(touching[["block_id", "building_count", "area_k"]].to_string())
print("within 600 m:", len(near), "blocks,", int(near.building_count.sum()), "bldgs")
bb = gpd.GeoSeries([win], crs=utm).to_crs(4326).total_bounds
roads = pyogrio.read_dataframe(
    "/home/gchurchill/.cache/reblock/osm_pbf/south-africa-latest.osm.pbf",
    layer="lines", bbox=tuple(bb), columns=["highway"])
roads = roads[roads.highway.notna()].to_crs(utm)
print(roads.highway.value_counts().to_string())
roads.to_parquet(f"{out}/roads5810.parquet")
near.to_parquet(f"{out}/blocks5810.parquet")
fig, ax = plt.subplots(figsize=(12, 12))
near.plot(ax=ax, column="building_count", cmap="Greys", alpha=0.6, edgecolor="k", lw=0.4)
gpd.GeoSeries([g], crs=utm).boundary.plot(ax=ax, color="red", lw=2)
major = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link",
         "primary_link", "secondary_link", "tertiary_link"}
roads[roads.highway.isin(major)].plot(ax=ax, color="blue", lw=2)
roads[roads.highway.isin({"residential", "unclassified", "living_street"})].plot(
    ax=ax, color="orange", lw=0.8)
roads[roads.highway.isin({"service", "track", "path", "footway"})].plot(
    ax=ax, color="green", lw=0.4)
for _, r in near[near.building_count >= 200].iterrows():
    p = r.geometry.representative_point()
    ax.annotate(f"{r.block_id.split('_')[-1]}\n{r.building_count}", (p.x, p.y),
                fontsize=7, ha="center")
x0, y0, x1, y1 = win.bounds
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
ax.set_axis_off()
fig.savefig(f"{out}/nbhd5810.png", dpi=110, bbox_inches="tight")
