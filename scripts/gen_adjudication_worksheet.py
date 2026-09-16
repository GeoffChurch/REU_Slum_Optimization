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

from scripts.gen_screen_bakeoff import METRICS, load

OUT = Path("data/adjudication")
UA = "reblock-research/0.1 (https://github.com/GeoffChurch/REU_Slum_Optimization)"
OB = Path.home() / ".cache/reblock/buildings_capetown_polygons.parquet"


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
    lab, cover = b["informal"].to_numpy(), b["cover"].to_numpy()
    short = {col: name.split("   ")[0] for col, name, _ in METRICS}
    ranks: dict[str, dict[int, int]] = {}
    for col, _, _ in METRICS:
        ranks[col] = {int(i): r for r, i in enumerate(np.argsort(-b[col].to_numpy())[:k], 1)}
    union = sorted(set().union(*(set(r) for r in ranks.values())))
    print(f"k={k}: {len(union)} blocks to adjudicate, from {k * len(METRICS)} screen-slots")

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

    wgs = Transformer.from_crs(b.crs, "EPSG:4326", always_xy=True)
    rows = []
    for i in union:
        c = b.geometry.iloc[i].centroid
        lon, latd = wgs.transform(c.x, c.y)
        n_ob, med = ob_med.get(i, (0, float("nan")))
        area_ha = float(b["a_m2"].iloc[i]) / 1e4
        rows.append({
            "block_id": b["block_id"].iloc[i],
            **{f"rank_{short[col]}": ranks[col].get(i, "") for col, _, _ in METRICS},
            "n_screens": sum(1 for col, _, _ in METRICS if i in ranks[col]),
            "buildings": int(b["building_count"].iloc[i]),
            "area_ha": round(area_ha, 2),
            "per_km2": round(int(b["building_count"].iloc[i]) / max(area_ha, 1e-9) * 100),
            "ob_n": n_ob, "ob_median_m2": "" if np.isnan(med) else round(med, 1),
            "survey_cover": round(float(cover[i]), 3),
            "survey_label": "informal" if lab[i] else "formal",
            "place": "", "maps": f"https://www.google.com/maps/@{latd:.5f},{lon:.5f},17z",
            "verdict": "", "notes": "",
        })

    print("reverse-geocoding (1 req/s, be patient)...", flush=True)
    for r in rows:
        lat, lon = r["maps"].split("@")[1].split(",")[:2]
        r["place"] = place_name(float(lat), float(lon))
        time.sleep(1.1)

    rows.sort(key=lambda r: (-int(r["n_screens"]),
                             min((int(v) for kk, v in r.items()
                                  if kk.startswith("rank_") and v != ""), default=99)))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"screen_top{k}_worksheet.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows; fill `verdict` + `notes`, leave survey_label alone)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
