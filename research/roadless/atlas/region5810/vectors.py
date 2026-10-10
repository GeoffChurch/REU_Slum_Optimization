"""Vector layers for the 5810@major page, in the 1 m rasters' frame (x right, y down, metres)."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely

HERE = Path("/home/gchurchill/src/reblock/research/roadless")
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OUT = Path(sys.argv[1])
fr = json.loads((OUT / "frame.json").read_text())
X0, Y1 = fr["x0"], fr["y0"] + fr["h"]
[b] = common.build_blocks(["ZAF.9.3.1_1_5810@major"])
polys = np.asarray(b.buildings.outlines)
f = b.boundary


def ring(coords):
    return " ".join(f"{x - X0:.1f} {Y1 - y:.1f}" for x, y in np.asarray(coords))


def poly_d(p):
    return " ".join("M" + ring(np.asarray(r.coords)[:-1]) + "Z"
                    for q in getattr(p, "geoms", [p]) for r in [q.exterior, *q.interiors])


def line_d(geoms):
    out = []
    for gm in geoms:
        for ln in getattr(gm, "geoms", [gm]):
            if ln.geom_type == "LineString" and ln.length > 0:
                out.append("M" + ring(ln.coords))
    return " ".join(out)


(OUT / "layers" / "buildings.json").write_text(json.dumps({"d": [poly_d(p) for p in polys]}))
allb = gpd.read_parquet("/home/gchurchill/.cache/reblock/blocks_capetown_full.parquet",
                        columns=["block_id", "geometry"]).to_crs(b.crs)
kb = allb[allb.intersects(f)]
bbox = tuple(gpd.GeoSeries([f.buffer(200)], crs=b.crs).to_crs(4326).total_bounds)
L = pyogrio.read_dataframe(common.REPO.parent / "x" if False else
                           "/home/gchurchill/.cache/reblock/osm_pbf/south-africa-latest.osm.pbf",
                           layer="lines", bbox=bbox, columns=["highway", "railway"]).to_crs(b.crs)
clip = f.buffer(150)
hw = L.highway.fillna("").str.removesuffix("_link")
sel = {"osm_major": hw.isin(["motorway", "trunk", "primary", "secondary", "tertiary"]),
       "osm_minor": hw.isin(["residential", "unclassified", "living_street", "service"]),
       "osm_path": hw.isin(["path", "footway", "track", "steps", "pedestrian"]),
       "osm_rail": L.railway.fillna("").isin(["rail"])}
ctx = {k: line_d(L[m].geometry.intersection(clip)) for k, m in sel.items()}
ctx["outline"] = "M" + ring(f.exterior.coords) + "Z"
ctx["kblocks"] = line_d([shapely.LineString(g.exterior.coords) if g.geom_type == "Polygon" else
                         shapely.MultiLineString([q.exterior.coords for q in g.geoms])
                         for g in kb.geometry.intersection(f)
                         if g.geom_type in ("Polygon", "MultiPolygon")])
b5810 = allb[allb.block_id.astype(str) == "ZAF.9.3.1_1_5810"]
ctx["s5810"] = "M" + ring(b5810.geometry.iloc[0].exterior.coords) + "Z"
(OUT / "layers" / "context.json").write_text(json.dumps(ctx))
print(len(polys), "buildings;", {k: len(v) // 1000 for k, v in ctx.items()}, "k chars")
