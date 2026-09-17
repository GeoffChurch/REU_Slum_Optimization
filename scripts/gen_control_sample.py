"""A RANDOM control sample, because the top-k worksheet cannot measure a rate.

`data/adjudication/screen_top15_worksheet.csv` holds 68 blocks that some screen ranked in its top
15. Every one is a screen's best guess, so the set is enriched for informal fabric by construction:
of the 35 adjudicated, 34 are informal and 1 is not. That set can show that survey misses EXIST --
it found 9 -- and it is the right set for ordering screens. It cannot estimate how OFTEN the survey
misses, and it cannot calibrate a judge, because a judge that answers "informal" every time scores
34/35 on it.

This draws from the whole eligible pool instead, stratified by the shipped screen's own rank.

## Why stratified rather than uniform

Informal blocks are 4.23% of the pool, and survey MISSES are rarer still. A uniform sample of 140
blocks would contain about 6 informal blocks and, at the rate the worksheet suggests, perhaps one
or two misses -- an estimate with an interval far wider than the quantity. Misses are not uniform:
they are new settlements, which the screen ranks HIGH and the 2018 survey does not know about, so
they concentrate in the top strata. Sampling proportional to where the signal is, and reweighting
by the known inclusion probability, buys precision for the same number of human judgements.

Each row carries `weight` = N_stratum / n_sampled_stratum. A Horvitz-Thompson total is
`sum(weight * indicator)` over adjudicated rows, and it is unbiased for the pool total whatever
the stratum sizes are -- that is the property uniform sampling would give up.

## What it is FOR, beyond the rate

Two things the worksheet cannot supply:

  1. Negatives. Calibrating an agent judge needs blocks that are genuinely NOT informal, and this
     is where they live.
  2. An honest denominator for "9 misses" -- a number that currently has no population attached.

`verdict` and `notes` are blank by design, exactly as in the worksheet, and `survey_label` is never
to be overwritten: the disagreement is the finding.

    pixi run python -m scripts.gen_control_sample            # 140 blocks, seed 20260917
    pixi run python -m scripts.gen_control_sample --n 60     # a smaller first pass
    pixi run python -m scripts.gen_control_sample --no-place # skip the Nominatim lookups
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from pyproj import Transformer

from reblock.data.counts import COUNTERS
from scripts.gen_adjudication_worksheet import _maps_url, place_name
from scripts.gen_screen_bakeoff import load

OUT = Path("data/adjudication/control_sample.csv")
SEED = 20260917
SCORE = "dd_proxy"          # the shipped screen -- strata are ITS rank, so misses concentrate high

# (label, upper rank fraction, share of the sample). Cut points are the retentions the bake-off
# already reports, so a stratum here means the same thing it means there.
STRATA: tuple[tuple[str, float, float], ...] = (
    ("top1pct",   0.01, 0.22),
    ("1to5pct",   0.05, 0.22),
    ("5to10pct",  0.10, 0.18),
    ("10to30pct", 0.30, 0.18),
    ("rest",      1.00, 0.20),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=140, help="total blocks to draw")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-place", action="store_true", help="skip Nominatim reverse geocoding")
    args = ap.parse_args()

    b, _ = load(COUNTERS["open_buildings"])
    score = b[SCORE].to_numpy()
    order = np.argsort(-score)                     # rank 0 = highest-scoring
    rank_of = np.empty(len(score), dtype=int)
    rank_of[order] = np.arange(len(score))
    n_pool = len(score)
    print(f"pool: {n_pool:,} eligible blocks, {int(b['informal'].sum()):,} survey-informal")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    lo_frac = 0.0
    for name, hi_frac, share in STRATA:
        lo, hi = int(lo_frac * n_pool), int(hi_frac * n_pool)
        members = order[lo:hi]
        take = min(len(members), max(1, round(args.n * share)))
        picked = rng.choice(members, size=take, replace=False)
        weight = len(members) / take
        print(f"  {name:<10} N={len(members):>6}  n={take:>3}  weight={weight:>7.1f}")
        for i in picked:
            rows.append({"block_id": str(b['block_id'].iloc[i]), "stratum": name,
                         "screen_rank": int(rank_of[i]) + 1, "n_stratum": len(members),
                         "n_sampled": take, "weight": round(weight, 4),
                         "dd_proxy": f"{score[i]:.6g}",
                         "ob_count": int(b['building_count'].iloc[i]),
                         "area_ha": f"{b['a_m2'].iloc[i] / 1e4:.2f}",
                         "survey_cover": f"{b['cover'].iloc[i]:.3f}",
                         "survey_label": "informal" if b['informal'].iloc[i] else "formal",
                         "_idx": int(i)})
        lo_frac = hi_frac

    rows.sort(key=lambda r: int(r["screen_rank"]))          # highest-scoring first
    to_wgs = Transformer.from_crs(b.crs, "EPSG:4326", always_xy=True)
    # `b` is projected (UTM 34S); `_maps_url` computes a zoom from a span in DEGREES, so the
    # geometry has to be reprojected and not merely its centroid. Reprojecting the whole frame
    # once beats a per-row transform of the ring coordinates.
    wgs = b.geometry.to_crs("EPSG:4326")
    for n, r in enumerate(rows, 1):
        idx = int(r.pop("_idx"))                            # type: ignore[call-overload]
        pt = b.geometry.iloc[idx].representative_point()
        lon, lat = to_wgs.transform(pt.x, pt.y)
        r["place"] = "" if args.no_place else place_name(lat, lon)
        r["maps"] = _maps_url(wgs.iloc[idx].envelope, lat, lon)
        r["verdict"] = ""
        r["notes"] = ""
        if not args.no_place:
            time.sleep(1.05)                                 # Nominatim: <= 1 request/second
            if n % 20 == 0:
                print(f"    geocoded {n}/{len(rows)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_inf = sum(1 for r in rows if r["survey_label"] == "informal")
    print(f"\nwrote {OUT}: {len(rows)} blocks, {n_inf} survey-informal, "
          f"{len(rows) - n_inf} survey-formal (the negatives a judge needs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
