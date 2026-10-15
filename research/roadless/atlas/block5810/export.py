"""5810's two answers at D 0.05 as map layers, on the gap complex (phase 2b; NOTES, "The complex as
the runtime: the ports and the cross-checks"): the flow before and after each preset's clearing
(gap_maps: the cells' current density |j| painted from the fixed mesh's triangles onto pixels of
H, times H, a pixel's crossing current; RGBA PNGs at one pixel each), buildings and context as SVG
paths, a JSON of numbers. One frame: metres, UTM. The arms are the complex's own presets on the
block (ARMS: their store runs, clearings.Store, under the runtime's metric)."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import shapely
from PIL import Image

HERE = Path("/home/gchurchill/src/reblock/research/roadless")
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402
import gap_maps  # noqa: E402
import metrics  # noqa: E402

BID = "ZAF.9.3.1_1_5810"
OUT = Path(sys.argv[1])
SP = Path(sys.argv[2])
PAD = 120.0
H = 0.5                  # a pixel's side (m): the page's layers, as the raster's cells were
PLAN = "fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2"
ARMS = {"cheap": ("cheap preset", f"polish_rows/{metrics.RUNTIME}/S0.01cat.P64w8/D0.05.p1"),
        "default": ("SIMP", f"relax_rows/{metrics.RUNTIME}/{PLAN}/D0.05.p1")}

[b] = common.build_blocks([BID])
metric = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU())
m = metric.clearing(b, demand=common.PopulationDemand())
geo = gc.geometry(b)
polys = np.asarray(b.buildings.outlines)
cost = m.cost
fr = gap_maps.Frame.around(geo, H, 0.0)
ny, nx = fr.ny, fr.nx
ox, oy = geo.origin
e = fr.extent
ext = [e[0] + ox, e[1] + ox, e[2] + oy, e[3] + oy]
frame = [ext[0] - PAD, ext[1] + PAD, ext[2] - PAD, ext[3] + PAD]
store = clearings.Store.committed()
clear = {key: np.asarray(store.cleared(run, b, step=1), dtype=int)
         for key, (_label, run) in ARMS.items()}


def rgba(v, cmap, norm, alpha):
    img = matplotlib.colormaps[cmap](norm(v))
    img[..., 3] = np.where(np.isnan(v), 0.0, alpha(v))
    return Image.fromarray((img[::-1] * 255).astype(np.uint8), "RGBA")   # row 0 = top


def fields(x):
    """(the crossing current per pixel, NaN off the open ground; the escape time per pixel)."""
    st = m.state(x)
    f = gap_maps.flow_of(m, st, m.pop)
    s = gap_maps.flow_image(m, st, f, fr) * H
    return s.ravel(), gap_maps.escape_image(m, st, f, fr).ravel(), float(m._rhs(st, m.pop) @ f.u)


s0, u0, P0 = fields(np.zeros(m.n))
on0 = np.isfinite(s0) & (s0 > 0)
top = float(np.quantile(s0[on0], 0.999))
lo = top * 1e-4
lognorm = mcolors.LogNorm(lo, top, clip=True)
flux_alpha = lambda v: np.clip((np.log10(np.maximum(np.nan_to_num(v), lo)) - np.log10(lo))  # noqa: E731
                               / np.log10(top / lo), 0, 1) ** 0.6
rgba(np.where(on0, s0, np.nan).reshape(ny, nx), "inferno", lognorm, flux_alpha) \
    .save(OUT / "layers" / "flux_before.png", optimize=True)
raw = {"s0": np.nan_to_num(s0).astype("float32"), "shape": np.array([ny, nx])}
info = {"block": BID, "metric": metric.name, "ext": ext, "frame": frame, "h": H,
        "n": int(len(polys)), "arms": {}, "sets": {}}
for key, (label, run) in ARMS.items():
    c = clear[key]
    x = np.zeros(m.n)
    x[c] = 1.0
    s1, u1, P1 = fields(x)
    on = on0 | (np.isfinite(s1) & (s1 > 0))
    rgba(np.where(np.isfinite(s1) & (s1 > 0), s1, np.nan).reshape(ny, nx), "inferno", lognorm,
         flux_alpha).save(OUT / "layers" / f"flux_{key}.png", optimize=True)
    d = np.where(on, np.nan_to_num(s1) - np.nan_to_num(s0), np.nan)
    lim = float(np.quantile(np.abs(d[on]), 0.995))
    sym = mcolors.SymLogNorm(lim * 1e-2, vmin=-lim, vmax=lim)
    rgba(d.reshape(ny, nx), "RdBu_r", sym,
         lambda v, sym=sym: np.clip(np.abs(np.nan_to_num(sym(v)) - 0.5) * 2, 0, 1) ** 0.5) \
        .save(OUT / "layers" / f"dflux_{key}.png", optimize=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ru = np.where(np.isfinite(u0) & (u0 > 0), (u1 - u0) / u0, np.nan)
    rn = mcolors.Normalize(-0.6, 0.6, clip=True)
    rgba(ru.reshape(ny, nx), "RdBu_r", rn,
         lambda v: np.clip(np.abs(np.nan_to_num(v)) / 0.3, 0, 1) ** 0.7 * 0.9) \
        .save(OUT / "layers" / f"escape_{key}.png", optimize=True)
    raw[f"s1_{key}"] = np.nan_to_num(s1).astype("float32")
    raw[f"ru_{key}"] = ru.astype("float32")
    raw["u0"] = u0.astype("float32")
    raw[f"u1_{key}"] = u1.astype("float32")
    info["arms"][key] = {"label": label, "run": run, "n": int(len(c)),
                         "D": float(cost[c].sum()), "P0": P0, "P1": P1,
                         "escape_median_change": float(np.nanmedian(ru)),
                         "faster_20pct_share": float(np.nanmean(ru < -0.2))}
    info["sets"][key] = sorted(int(i) for i in c)
    print(key, {k: v for k, v in info["arms"][key].items() if k != "run"}, flush=True)

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
