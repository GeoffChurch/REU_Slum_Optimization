"""How much of Cape Town the major-road faces cover (the denominator for item 6's
extrapolation): the faces of regions.MAJOR's network (motorway to tertiary, links included)
over the Cape Town kblock source's extent, each face's kblocks (by representative point, as
regions.build takes them) and their building counts.

    ... uv run python research/roadless/fft_homog/region_capetown_faces.py
Writes OUT/capetown_faces.csv (one row per closed face holding a kblock).
"""
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
import region_common as rc  # noqa: E402
import regions  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import box  # noqa: E402
from shapely.ops import polygonize, unary_union  # noqa: E402

from reblock.data.pools import pbf_path  # noqa: E402

with rc.Monitor("Cape Town major-road faces (polygonize, kblock counts)"):
    t = time.time()
    blocks = gpd.read_parquet(Path.home() / ".cache/reblock/blocks_capetown_full.parquet",
                              columns=["block_id", "building_count", "block_area_m2",
                                       "geometry"])
    crs = blocks.estimate_utm_crs()
    blocks = blocks.to_crs(crs)
    x0, y0, x1, y1 = blocks.total_bounds
    frame = box(x0 - 2000, y0 - 2000, x1 + 2000, y1 + 2000)
    bbox = tuple(gpd.GeoSeries([frame], crs=crs).to_crs(4326).total_bounds)
    ways = pyogrio.read_dataframe(pbf_path("ZAF"), layer="lines", bbox=bbox,
                                  columns=["highway"]).to_crs(crs)
    ways = ways[ways.highway.fillna("").str.removesuffix("_link").isin(regions.MAJOR.classes)]
    print(f"{len(blocks)} kblocks, {len(ways)} major ways, {time.time() - t:.0f} s", flush=True)
    faces = [f for f in polygonize(unary_union([*ways.geometry, frame.exterior]))
             if f.boundary.distance(frame.exterior) > 1.0]
    faces = np.array([shapely.Polygon(f.exterior) for f in faces], dtype=object)
    print(f"{len(faces)} closed faces, {time.time() - t:.0f} s", flush=True)
    pts = shapely.points(np.c_[blocks.geometry.representative_point().x,
                               blocks.geometry.representative_point().y])
    tree = shapely.STRtree(faces)
    bi, fi = tree.query(pts, predicate="within")
    # nested faces cannot occur (a face has no holes here); a point on a shared edge goes to one
    first = pd.Series(fi, index=bi).groupby(level=0).first()
    df = pd.DataFrame(dict(face=first.to_numpy(),
                           buildings=blocks.building_count.to_numpy()[first.index],
                           kblock_area_m2=blocks.block_area_m2.to_numpy()[first.index]))
    g = df.groupby("face").agg(kblocks=("buildings", "size"), buildings=("buildings", "sum"),
                               kblock_area_m2=("kblock_area_m2", "sum"))
    g["face_area_m2"] = shapely.area(faces[g.index.to_numpy()])
    g.to_csv(rc.OUT / "capetown_faces.csv")
    for floor in (1, 10):
        h = g[g.buildings >= floor]
        print(f"faces with >= {floor} buildings: {len(h)}, {h.face_area_m2.sum() / 1e6:.0f} km^2 "
              f"(median {h.face_area_m2.median() / 1e6:.3f}, p90 "
              f"{np.percentile(h.face_area_m2, 90) / 1e6:.2f}, max {h.face_area_m2.max() / 1e6:.1f}"
              f" km^2), {int(h.buildings.sum())} buildings", flush=True)
    lost = np.setdiff1d(np.arange(len(blocks)), first.index)
    print(f"kblocks in no closed face: {len(lost)} "
          f"({blocks.building_count.to_numpy()[lost].sum()} buildings)", flush=True)
