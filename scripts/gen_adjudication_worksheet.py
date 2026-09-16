"""The blocks a human must look at to ORDER the screens at top-k, and nothing else.

Why this set. `precision@k` differences between two screens are `(1/k) *` the sum over their
SYMMETRIC DIFFERENCE -- any block both rank cancels exactly -- so only blocks some screens rank
and others do not carry ordering information. Measured on Cape Town the shared core is empty at
k <= 15 (the screens disagree that strongly at the top), so the union IS the adjudication set:
41 blocks at k=15, against 60 screen-slots.

Why k=15. k=10 is the smallest k that separates all four screens on survey labels, but its
tightest margin is ONE block. k=15 widens the margin on the claim that matters -- the shipped
screen is best -- to three, for eleven more blocks. Both the top (`depth_density proxy`) and the
bottom (`density`) are stable at every k from 3 to 30; the middle two genuinely CROSS between
k=15 and k=30, so they are a tie to be reported, not an order to be bought.

Deliberately NOT triaged by Open Buildings footprint size. It would cut the set several-fold,
but OB morphology correlates rho 0.5-0.77 with the density metrics being judged
(notes/2026-08-09-c16-c18-...), so letting it choose which blocks a human sees reintroduces that
circularity through the back door. The `ob_median_m2` column is present as CONTEXT for the
adjudicator, not as a filter.

`verdict` and `notes` are blank by design: fill them in, commit, and the file becomes the second
label column. Never overwrite `survey_label` -- the disagreements are the finding.

    pixi run python -m scripts.gen_adjudication_worksheet [k]
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely import STRtree

from reblock.data.counts import BuildingCount, KblockCount, OpenBuildingsCount, resolved
from reblock.screen.dense_compact import _chunk_depths
from scripts.gen_screen_bakeoff import METRICS, load

OUT = Path("data/adjudication")
UA = "reblock-research/0.1 (https://github.com/GeoffChurch/REU_Slum_Optimization)"
OB = Path.home() / ".cache/reblock/buildings_capetown_polygons.parquet"



def _maps_url(geom: object, lat: float, lon: float, viewport_px: int = 900) -> str:
    """A satellite link ZOOMED TO THE BLOCK, not to a fixed level.

    A fixed zoom is wrong at both ends of this worksheet: `ZAF.9.3.1_1_44685` is 0.3 ha and
    `_5810` is 58 ha, and 17z shows one as a speck and crops the other. Web Mercator gives
    metres-per-pixel as `156543.03392 * cos(lat) / 2^z`, so the zoom that fits an extent of
    `E` metres across `viewport_px` is `log2(156543.03392 * cos(lat) * viewport_px / E)`.

    `/data=!3m1!1e3` opens in satellite rather than the road map -- what the judgement is
    actually made on.
    """
    import math

    x0, y0, x1, y1 = geom.bounds                          # type: ignore[attr-defined]
    span_m = max((x1 - x0) * 111_320 * math.cos(math.radians(lat)), (y1 - y0) * 110_540)
    z = math.log2(156543.03392 * math.cos(math.radians(lat)) * viewport_px / max(span_m, 1.0))
    return (f"https://www.google.com/maps/@{lat:.5f},{lon:.5f},"
            f"{max(14.0, min(19.5, z)):.1f}z/data=!3m1!1e3")


def place_name(lat: float, lon: float) -> str:
    """Most specific OSM place name at this point, or "" -- a starting point for the human,
    not an authority. Nominatim asks for <= 1 request/second and a real User-Agent; both are
    honoured, and any failure degrades to an empty name rather than stopping the run."""
    q = urllib.parse.urlencode({"format": "json", "lat": f"{lat:.5f}", "lon": f"{lon:.5f}",
                                "zoom": "16", "addressdetails": "1"})
    try:
        req = urllib.request.Request(f"https://nominatim.openstreetmap.org/reverse?{q}",
                                     headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:      # noqa: S310 (fixed https host)
            addr = json.load(r).get("address", {})
    except Exception:                                            # noqa: BLE001 (offline is fine)
        return ""
    for key in ("neighbourhood", "suburb", "city_district", "town", "village"):
        if addr.get(key):
            return str(addr[key])
    return ""


def main() -> int:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    b, _ = load()
    b = b.rename(columns={"a_m2": "block_area_m2"})
    lab, cover = b["informal"].to_numpy(), b["cover"].to_numpy()
    short = {col: name.split("   ")[0] for col, name, _ in METRICS}
    pts = str(Path.home() / ".cache/reblock/buildings_capetown_full.parquet")

    # BOTH count sources. Switching the default to Open Buildings changes WHICH blocks the screens
    # select, and the blocks it adds are exactly the ones in dispute -- a worksheet built on one
    # source would spend an afternoon at the wrong 41 blocks.
    sources: dict[str, BuildingCount] = {"kb": KblockCount(), "ob": OpenBuildingsCount()}
    counts: dict[str, np.ndarray] = {}
    ranks: dict[str, dict[int, int]] = {}
    for tag, counter in sources.items():
        r = resolved(b, pts, counter)
        n = r["building_count"].to_numpy(dtype=float)
        counts[tag] = n
        a = r["block_area_m2"].to_numpy(dtype=float)
        per = r.geometry.length.to_numpy()
        for col, _, _ in METRICS:
            sc = {"depth_density proxy": np.sqrt(n * a) / per * (n / a), "density": n / a,
                  "density_compactness": n / per ** 2,
                  "depth proxy": np.sqrt(n * a) / per}[short[col]]
            ranks[f"{tag}:{col}"] = {int(i): rr for rr, i in enumerate(np.argsort(-sc)[:k], 1)}
    union = sorted(set().union(*(set(r) for r in ranks.values())))
    in_kb = {i for key, r in ranks.items() if key.startswith("kb") for i in r}
    in_ob = {i for key, r in ranks.items() if key.startswith("ob") for i in r}
    print(f"k={k}: {len(union)} blocks to adjudicate "
          f"(kblock-union {len(in_kb)}, OB-union {len(in_ob)}, shared {len(in_kb & in_ob)})")

    ob_med: dict[int, tuple[int, float]] = {}
    if OB.exists():
        polys = gpd.read_parquet(OB).to_crs(b.crs)
        areas = polys.geometry.area.to_numpy()
        tree = STRtree(list(polys.geometry.centroid))
        for i in union:
            # `contains`, not `within`: STRtree applies `query_geom.predicate(tree_geom)`, so
            # `within` asks whether the BLOCK is inside a centroid and silently returns nothing.
            hit = tree.query(b.geometry.iloc[i], predicate="contains")
            ob_med[i] = (len(hit), float(np.median(areas[hit])) if len(hit) else float("nan"))

    # The FINE metric -- real Voronoi tessellation + BFS peel -- on exactly these blocks. The
    # bake-off is proxy-only by design ("no Voronoi, no peel"), which is how `ZAF.9.3.1_1_5810`
    # sat at 8,887th by proxy while being first by true depth. ~3 s/block, so affordable here and
    # nowhere near affordable over a metro, which is the whole reason proxies exist.
    print(f"fine pass (Voronoi + peel) on {len(union)} blocks...", flush=True)
    cache = Path.home() / ".cache/reblock"
    fine = dict(_chunk_depths((str(cache / "blocks_capetown_full.parquet"),
                               str(cache / "buildings_capetown_full.parquet"), 10,
                               [str(b["block_id"].iloc[i]) for i in union])))

    wgs = Transformer.from_crs(b.crs, "EPSG:4326", always_xy=True)
    wgs_geoms = b.to_crs("EPSG:4326").geometry
    rows = []
    for i in union:
        c = b.geometry.iloc[i].centroid
        lon, latd = wgs.transform(c.x, c.y)
        geom_wgs = wgs_geoms.iloc[i]
        n_ob, med = ob_med.get(i, (0, float("nan")))
        area_ha = float(b["block_area_m2"].iloc[i]) / 1e4
        rows.append({
            "block_id": b["block_id"].iloc[i],
            "disputed": "yes" if (i in in_ob) != (i in in_kb) else "",
            **{f"{tag}_{short[col]}": ranks[f"{tag}:{col}"].get(i, "")
               for tag in sources for col, _, _ in METRICS},
            "n_screens": sum(1 for key in ranks if i in ranks[key]),
            "kb_count": int(counts["kb"][i]), "ob_count": int(counts["ob"][i]),
            "ob_over_kb": round(counts["ob"][i] / max(counts["kb"][i], 1.0), 2),
            "area_ha": round(area_ha, 2),
            "per_km2": round(counts["ob"][i] / max(area_ha, 1e-9) * 100),
            "ob_n": n_ob, "ob_median_m2": "" if np.isnan(med) else round(med, 1),
            "fine_depth": fine.get(str(b["block_id"].iloc[i]), ""),
            "fine_depth_density": "",      # filled below, once every block's fine depth is known
            "survey_cover": round(float(cover[i]), 3),
            "survey_label": "informal" if lab[i] else "formal",
            "place": "", "maps": _maps_url(geom_wgs, latd, lon),
            "verdict": "", "notes": "",
        })

    print("reverse-geocoding (1 req/s, be patient)...", flush=True)
    for r in rows:
        lat, lon = r["maps"].split("@")[1].split(",")[:2]
        r["place"] = place_name(float(lat), float(lon))
        time.sleep(1.1)

    # `fine_depth_density` = the shipped screen's own FINE form, depth x density, on Open
    # Buildings counts. Closer to "is this an informal settlement" than depth alone, which
    # measures nesting: a gated estate of cul-de-sacs is deep and is not a settlement.
    for r in rows:
        d = float(r["fine_depth"]) if r["fine_depth"] != "" else 0.0
        r["fine_depth_density"] = round(d * float(r["per_km2"]) / 1e4, 3)

    # `disagreement` = how far the best automated signal is from the survey's verdict, in
    # percentile terms: a formal-labelled block scoring HIGH disagrees, an informal-labelled
    # block scoring LOW disagrees. It orders the adjudication; it does not decide anything.
    fdd = np.array([float(r["fine_depth_density"]) for r in rows])
    pct = fdd.argsort().argsort() / max(len(fdd) - 1, 1)
    for r, q in zip(rows, pct, strict=True):
        r["disagreement"] = round(q if r["survey_label"] == "formal" else 1.0 - q, 3)

    # Disputed first -- those decide whether the Open Buildings switch is right, which is what is
    # blocked on this pass -- then by disagreement, so within the critical path the blocks where
    # the fine metric and the survey conflict most come first.
    rows.sort(key=lambda r: (r["disputed"] != "yes", -float(r["disagreement"])))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"screen_top{k}_worksheet.csv"

    # Carry forward any adjudication already done. Regenerating is routine -- a changed k, a new
    # count source, a better link -- and a regeneration that silently discarded an afternoon of
    # verdicts would be the worst possible failure of this file. Keyed on block_id, so rows that
    # move or leave the top-k keep their verdict if they come back.
    if path.exists():
        prior = {r["block_id"]: (r.get("verdict", ""), r.get("notes", ""))
                 for r in csv.DictReader(path.open()) if r.get("verdict", "").strip()}
        carried = 0
        for r in rows:
            if r["block_id"] in prior:
                r["verdict"], r["notes"] = prior[r["block_id"]]
                carried += 1
        if prior:
            print(f"carried forward {carried}/{len(prior)} existing verdict(s)"
                  + (f"; {len(prior) - carried} no longer in the top-{k} union"
                     if carried < len(prior) else ""))

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows; fill `verdict` + `notes`, leave survey_label alone)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
