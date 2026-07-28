"""How many eligible donors does a typical covered recipient actually have?

The prediction branch ("reconstruct a block's real footpaths from the consensus of its mapped
neighbours") reached 94% of a block's own OSM on 2026-07-23 -- at n=1, in well-mapped Cape Town,
and even there 8 of the 15 most-similar neighbours had ZERO interior OSM. The 2026-07-28 census
put national coverage at 25.2%, so before scaling that result the question is whether a typical
recipient has enough *nearby, mapped, non-leaking* donors to form a consensus at all.

Two constraints pull against each other:
  * leakage says donors must be FAR -- neighbouring blocks share a mapper session and often the
    same OSM way clipped at a block edge, so `exclusion_holdout(radius_m)` is the fold definition.
  * similarity says donors should be NEAR -- morphology is regional, and a donor from another
    metro is a worse match.

So the number that matters is not "how many donors exist" but the DISTANCE TO THE k-TH NEAREST
eligible donor, and how much the exclusion radius costs.

Distances are chord distances on a sphere (lon/lat -> ECEF, KD-tree). At these radii the chord
underestimates the great-circle arc by <1e-5 relative -- far below anything this decides.

    pixi run python -m scripts.donor_availability
    pixi run python -m scripts.donor_availability --min-density 500 --exclusion-m 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from scipy.spatial import cKDTree

CACHE = Path.home() / ".cache" / "reblock"
ISOS = ("ZAF", "KEN")
EARTH_R_M = 6_371_008.8
K_VALUES = (1, 5, 15)
RADII_KM = (2, 5, 10, 20, 50)


def candidates(min_density: float, min_k: int, min_buildings: int,
               max_buildings: int) -> pd.DataFrame:
    """Census rows that clear the qualified band AND the density floor.

    The density floor is the 2026-07-28 census's finding: a building-count band does not bound
    block AREA, so outside a metro the "qualified" pool fills up with rural district polygons
    (median 1.12 km2, max 2,000 km2) whose interior footpaths are a whole region's.
    """
    frames = []
    for iso in ISOS:
        path = CACHE / f"osm_coverage_{iso}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path} -- run `python -m scripts.osm_census --iso {iso}`")
        frames.append(pd.read_parquet(path).assign(iso=iso))
    df = pd.concat(frames, ignore_index=True)
    df = df[~df["census_failed"]].copy()
    df["density"] = df["building_count"] / (df["area_m2"] / 1e6)
    keep = (df["building_count"].between(min_buildings, max_buildings)
            & (df["k_complexity"] >= min_k)
            & (df["density"] >= min_density))
    return df[keep].reset_index(drop=True)


def centroids(block_ids: set[str]) -> dict[str, tuple[float, float]]:
    """(lon, lat) of each wanted block, streamed out of the country parquets.

    Same GeoParquet-1.0 decode as the census (`osm_census._decode_batch`): geometry is a plain
    Arrow binary field whose `geo` metadata is file-level, so `from_arrow` cannot see it.
    """
    out: dict[str, tuple[float, float]] = {}
    for iso in ISOS:
        pf = pq.ParquetFile(CACHE / f"{iso}_geodata.parquet")
        for batch in pf.iter_batches(batch_size=50_000, columns=["block_id", "geometry"]):
            frame = batch.to_pandas()
            frame["block_id"] = frame["block_id"].astype(str)
            hit = frame[frame["block_id"].isin(block_ids)]
            if hit.empty:
                continue
            geom = shapely.from_wkb(hit["geometry"])
            pts = shapely.point_on_surface(geom)
            for bid, x, y in zip(hit["block_id"], shapely.get_x(pts), shapely.get_y(pts),
                                 strict=True):
                out[bid] = (float(x), float(y))
    return out


def _ecef(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lo, la = np.radians(lon), np.radians(lat)
    return EARTH_R_M * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-density", type=float, default=1000.0)
    ap.add_argument("--min-k", type=int, default=4)
    ap.add_argument("--min-buildings", type=int, default=60)
    ap.add_argument("--max-buildings", type=int, default=300)
    ap.add_argument("--exclusion-m", type=float, default=2000.0,
                    help="donors at or below this distance are held out as same-settlement")
    ap.add_argument("--min-interior-m", type=float, default=100.0,
                    help="a donor with a few metres of footpath is not donatable material")
    args = ap.parse_args()

    pool = candidates(args.min_density, args.min_k, args.min_buildings, args.max_buildings)
    donor_ok = pool["interior_length_m_0.5"] >= args.min_interior_m
    print(f"qualified + density>={args.min_density:.0f}/km2 : {len(pool):,} blocks")
    n_cov = int((pool["n_interior_segments_0.5"] > 0).sum())
    print(f"  covered (>=1 interior segment)          : {n_cov:,}")
    print(f"  donatable (>={args.min_interior_m:.0f} m interior)          : {donor_ok.sum():,}")

    xy = centroids(set(pool["block_id"].astype(str)))
    pool = pool[pool["block_id"].astype(str).isin(xy)].reset_index(drop=True)
    lon = np.array([xy[str(b)][0] for b in pool["block_id"]])
    lat = np.array([xy[str(b)][1] for b in pool["block_id"]])
    pts = _ecef(lon, lat)
    donor_ok = (pool["interior_length_m_0.5"] >= args.min_interior_m).to_numpy()
    recipients = np.flatnonzero((pool["n_interior_segments_0.5"] > 0).to_numpy())
    donors = np.flatnonzero(donor_ok)
    print(f"\nlocated {len(pool):,} blocks; {len(recipients):,} recipients, "
          f"{len(donors):,} donors")

    tree = cKDTree(pts[donors])
    # k+1: a recipient that is itself donatable appears in its own donor tree.
    kmax = max(K_VALUES) + 1
    dist, _idx = tree.query(pts[recipients], k=min(kmax, len(donors)))
    dist = np.atleast_2d(dist)

    print("\n--- distance to the k-th nearest donor, NO exclusion (km) ---")
    for k in K_VALUES:
        if k >= dist.shape[1]:
            continue
        d = np.sort(dist, axis=1)[:, k] / 1000.0          # column k skips self at column 0
        print(f"  k={k:2d}: median {np.median(d):7.2f}  p90 {np.quantile(d, .9):8.2f}  "
              f"within 20 km: {(d <= 20).mean():5.1%}")

    # What the holdout costs the 2026-07-23 recipe specifically. That study picked its consensus
    # donors by morphological similarity from a small pool, with no distance constraint at all --
    # so the share of a recipient's nearest donors that the exclusion radius removes is a direct
    # estimate of how much of that 94% was neighbours-share-a-mapper leakage.
    k_consensus = max(K_VALUES)
    if dist.shape[1] > k_consensus:
        nearest = np.sort(dist, axis=1)[:, 1:k_consensus + 1]
        leaking = (nearest <= args.exclusion_m).mean(axis=1)
        print(f"\n--- of a recipient's nearest {k_consensus} donors, the share inside the "
              f"{args.exclusion_m:.0f} m exclusion radius ---")
        print(f"  median {np.median(leaking):.1%}   mean {leaking.mean():.1%}   "
              f"recipients where ALL {k_consensus} leak: {(leaking == 1.0).mean():.1%}")

    print(f"\n--- with the {args.exclusion_m:.0f} m exclusion radius applied ---")
    counts = tree.query_ball_point(pts[recipients], r=args.exclusion_m,
                                   return_length=True)
    print(f"  donors INSIDE the exclusion radius (leakage risk), median: "
          f"{np.median(counts):.0f}")
    for radius_km in RADII_KM:
        n_in = tree.query_ball_point(pts[recipients], r=radius_km * 1000.0,
                                     return_length=True)
        eligible = n_in - counts
        for k in K_VALUES:
            frac = (eligible >= k).mean()
            print(f"  within {radius_km:2d} km: >= {k:2d} eligible donors for {frac:6.1%} "
                  f"of recipients", end="")
            if k == K_VALUES[-1]:
                print(f"   (median eligible: {np.median(eligible):.0f})")
            else:
                print()


if __name__ == "__main__":
    main()
