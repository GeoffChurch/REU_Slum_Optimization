"""5810's two answers at D 0.05 as map layers: flow before/after per clearing (RGBA PNGs at one
pixel a cell), buildings and context as SVG paths, a JSON of numbers. One frame: metres, UTM."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import shapely
from PIL import Image

HERE = Path("/home/gchurchill/src/reblock/research/roadless")
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from current_map import flow  # noqa: E402

BID = "ZAF.9.3.1_1_5810"
OUT = Path(sys.argv[1])
SP = Path(sys.argv[2])
PAD = 120.0
ARMS = {"cheap": "cheap preset", "default": "SIMP"}

[b] = common.build_blocks([BID])
p = lifted.Params(3.0, 8, solver=lifted.solver_of("gpu"))
sc = common.Scorer(b, lifted.UniformMesh(0.5, offset=lifted.OFFSET), p,
                   population=common.POPULATIONS["area"])
g, polys = sc.grid, sc.polys
lab = g.label_sub(polys)
cost = sc.w / sc.w.sum()
ny, nx = g.inside.shape
ext = [g.x0 - g.h / 2, g.x0 + (nx - 0.5) * g.h, g.y0 - g.h / 2, g.y0 + (ny - 0.5) * g.h]
frame = [ext[0] - PAD, ext[1] + PAD, ext[2] - PAD, ext[3] + PAD]
clear = pd.read_parquet(HERE / "fine_clearings.parquet")
clear = {n: np.asarray(c, dtype=int) for n, c in
         zip(clear[clear.block == BID].name, clear[clear.block == BID].cleared, strict=True)}


def rgba(v, cmap, norm, alpha):
    img = matplotlib.colormaps[cmap](norm(v))
    img[..., 3] = np.where(np.isnan(v), 0.0, alpha(v))
    return Image.fromarray((img[::-1] * 255).astype(np.uint8), "RGBA")   # row 0 = top


def S(f):       # current through each cell: sum over headings of |signed flux|
    return np.abs(f.phi).sum(axis=0)


f0 = flow(sc, g.ff0, p)
s0 = S(f0)
on0 = s0 > 0
top = float(np.quantile(s0[on0], 0.999))
lo = top * 1e-4
lognorm = mcolors.LogNorm(lo, top, clip=True)
flux_alpha = lambda v: np.clip((np.log10(np.maximum(np.nan_to_num(v), lo)) - np.log10(lo))  # noqa: E731
                               / np.log10(top / lo), 0, 1) ** 0.6
rgba(np.where(on0, s0, np.nan).reshape(ny, nx), "inferno", lognorm, flux_alpha) \
    .save(OUT / "layers" / "flux_before.png", optimize=True)
raw = {"s0": s0.astype("float32"), "shape": np.array([ny, nx])}
info = {"block": BID, "ext": ext, "frame": frame, "h": g.h, "n": int(len(polys)), "arms": {}}
for key, label in ARMS.items():
    c = clear[key]
    after = (g.isub & (~g.bsub | np.isin(lab, c))).mean(axis=-1)
    f1 = flow(sc, after, p)
    s1 = S(f1)
    on = on0 | (s1 > 0)
    rgba(np.where(s1 > 0, s1, np.nan).reshape(ny, nx), "inferno", lognorm, flux_alpha) \
        .save(OUT / "layers" / f"flux_{key}.png", optimize=True)
    d = np.where(on, s1 - s0, np.nan)
    lim = float(np.quantile(np.abs(d[on]), 0.995))
    sym = mcolors.SymLogNorm(lim * 1e-2, vmin=-lim, vmax=lim)
    rgba(d.reshape(ny, nx), "RdBu_r", sym,
         lambda v, sym=sym: np.clip(np.abs(np.nan_to_num(sym(v)) - 0.5) * 2, 0, 1) ** 0.5) \
        .save(OUT / "layers" / f"dflux_{key}.png", optimize=True)
    was = g.ff0.ravel() > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ru = np.where(was & (f0.u > 0), (f1.u - f0.u) / f0.u, np.nan)
    rn = mcolors.Normalize(-0.6, 0.6, clip=True)
    rgba(ru.reshape(ny, nx), "RdBu_r", rn,
         lambda v: np.clip(np.abs(np.nan_to_num(v)) / 0.3, 0, 1) ** 0.7 * 0.9) \
        .save(OUT / "layers" / f"escape_{key}.png", optimize=True)
    raw[f"s1_{key}"] = s1.astype("float32")
    raw[f"ru_{key}"] = ru.astype("float32")
    raw["u0"] = f0.u.astype("float32")
    raw[f"u1_{key}"] = f1.u.astype("float32")
    info["arms"][key] = {"label": label, "n": int(len(c)), "D": float(cost[c].sum()),
                         "P0": float(f0.P), "P1": float(f1.P),
                         "escape_median_change": float(np.nanmedian(ru)),
                         "faster_20pct_share": float(np.nanmean(ru < -0.2))}
    print(key, info["arms"][key], flush=True)

# vectors: y flipped into SVG's frame (y down), 0.1 m precision
X0, Y1 = frame[0], frame[3]


def path(geoms):
    out = []
    for gm in geoms:
        for poly in getattr(gm, "geoms", [gm]):
            for ring in [poly.exterior, *poly.interiors] if poly.geom_type == "Polygon" else [poly]:
                xy = np.asarray(ring.coords)
                pts = " ".join(f"{x - X0:.1f} {Y1 - y:.1f}" for x, y in xy)
                out.append("M" + pts + ("Z" if poly.geom_type == "Polygon" else ""))
    return " ".join(out)


sets = {k: set(v.tolist()) for k, v in clear.items()}
both = sets["cheap"] & sets["default"]
groups = {"kept": [i for i in range(len(polys)) if i not in sets["cheap"] | sets["default"]],
          "both": sorted(both), "cheap_only": sorted(sets["cheap"] - both),
          "simp_only": sorted(sets["default"] - both)}
svg = {k: path(polys[v]) for k, v in groups.items()}
svg["outline"] = path([b.boundary.exterior, *[shapely.LineString(r) for r in b.boundary.interiors]])
box = shapely.box(frame[0], frame[2], frame[1], frame[3])
nb = gpd.read_parquet(SP / "region" / "blocks5810.parquet")
nb = nb[(nb.block_id != BID) & nb.intersects(box)]
svg["neighbours"] = path([shapely.LineString(r.exterior.coords)
                          for r in nb.geometry.intersection(box)
                          if r.geom_type == "Polygon"])
x0, y0, x1, y1 = gpd.GeoSeries([box], crs=nb.crs).to_crs(4326).total_bounds
bld = gpd.read_parquet("/home/gchurchill/.cache/reblock/buildings_capetown_polygons.parquet",
                       columns=["geometry"])
bld = bld.cx[x0:x1, y0:y1].to_crs(nb.crs)
bld = bld[bld.intersects(box) & ~bld.within(b.boundary)]
svg["context_buildings"] = path(list(bld.geometry))
roads = gpd.read_parquet(SP / "region" / "roads5810.parquet")
roads = roads[roads.intersects(box)]
for cls, keys in {"osm_major": ["motorway", "trunk", "primary", "secondary", "tertiary"],
                  "osm_minor": ["residential", "unclassified", "living_street", "service"],
                  "osm_path": ["path", "footway", "track", "steps", "pedestrian"]}.items():
    sel = roads[roads.highway.str.replace("_link", "").isin(keys)].geometry.intersection(box)
    svg[cls] = path([x for gm in sel for x in getattr(gm, "geoms", [gm]) if x.length > 0])
info["groups"] = {k: len(v) for k, v in groups.items()}
info["svg_size"] = [frame[1] - frame[0], frame[3] - frame[2]]
np.savez_compressed(OUT / "raw.npz", **raw)
(OUT / "layers" / "vectors.json").write_text(json.dumps(svg))
(OUT / "layers" / "info.json").write_text(json.dumps(info, indent=1))
print({k: len(v) // 1000 for k, v in svg.items()}, "k chars")
