"""f(k) = the smallest `proxy_keep_n` that retains the fine metric's top-k. The whole curve.

`scripts/proxy_keep_retention.py` reports one point of this, f(15). This is the function.

## What it is

Rank every eligible block twice -- once by the expensive `metric.fine`, once by a cheap proxy --
and let `sigma(j)` be the PROXY rank of the block the fine metric ranks j-th. Then

    f(k) = max over j <= k of sigma(j)          numpy: np.maximum.accumulate(sigma)

is the cumulative (prefix) maximum of the rank permutation. `f` is non-decreasing, `f(k) >= k`
always, and `f(N) = N` necessarily -- the last block forces it.

Two pieces of standard structure are worth naming, because both mean something here:

* `f` JUMPS exactly at the left-to-right maxima (records) of `sigma`. A long flat stretch is a run
  of fine-ranked blocks the proxy already had inside the prefix; a jump is one block the proxy
  buried, and the height is how deep.
* `f(k) = k` exactly when `sigma` maps {1..k} onto {1..k} -- the prefix is CLOSED. In permutation
  terms these are the direct-sum decomposition points, and the number of them is the number of
  indecomposable components. Here they are the SAFE CUT POINTS: the only values of
  `proxy_keep_n` that retain the fine top-k with nothing to spare.

The curve is the honest form of the pre-filter question, because a single f(15) hides whether you
are one unlucky block from a cliff. `f(15) = 106` with `f(16) = 4,000` is a very different setting
from `f(15) = 106` with `f(200) = 250`.

    pixi run python -m scripts.retention_curve
    pixi run python -m scripts.retention_curve --cities capetown --png out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from numpy.typing import NDArray
from shapely import STRtree

from reblock.data.counts import resolved

KS = (1, 5, 15, 50, 100, 500, 1000, 5000)
VARIANTS = ("depth", "depth_density")
UNBOUNDED = 10**9


def curve(fine_order: list[str], proxy_rank: dict[str, int]) -> NDArray[np.int64]:
    """The prefix maximum of the rank permutation -- f(k) for every k, in one pass."""
    sigma = np.fromiter((proxy_rank[b] for b in fine_order), dtype=np.int64, count=len(fine_order))
    return np.maximum.accumulate(sigma)


def _p90(blocks: gpd.GeoDataFrame, buildings_path: str) -> NDArray[np.float64]:
    g = gpd.read_parquet(buildings_path, columns=["area_in_meters", "geometry"]).to_crs(blocks.crs)
    a = g["area_in_meters"].to_numpy()
    blk, pt = STRtree(list(g.geometry)).query(np.asarray(blocks.geometry), predicate="contains")
    o = np.argsort(blk)
    blk, pt = blk[o], pt[o]
    lo = np.searchsorted(blk, np.arange(len(blocks)), "left")
    hi = np.searchsorted(blk, np.arange(len(blocks)), "right")
    return np.array([np.percentile(a[pt[lo[i]:hi[i]]], 90) if hi[i] > lo[i] else np.nan
                     for i in range(len(blocks))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", default="capetown,nairobi")
    ap.add_argument("--png", default="")
    args = ap.parse_args()

    series: dict[str, NDArray[np.int64]] = {}
    for city in args.cities.split(","):
        for variant in VARIANTS:
            with initialize_config_dir(version_base=None,
                                       config_dir=str(Path("conf").resolve())):
                cfg = compose(config_name="compare_config",
                              overrides=[f"+example={variant}", f"data={city}_full",
                                         f"proxy_keep_n={UNBOUNDED}"])
            source, screen = instantiate(cfg.data), instantiate(cfg.screen)
            fine_order = list(screen.select(source))
            bl = gpd.read_parquet(source.blocks_path,
                                  columns=["block_id", "building_count", "block_area_m2",
                                           "geometry"])
            bl = resolved(bl, source.buildings_path, screen.counts)
            bid = bl["block_id"].astype(str).to_numpy()
            n = bl["building_count"].to_numpy(dtype=float)
            a = bl["block_area_m2"].to_numpy(dtype=float)
            p = bl.geometry.length.to_numpy()
            p90 = _p90(bl, str(source.buildings_path))
            elig = n >= screen.min_buildings
            for nm, sc in {"dd_proxy": n**1.5 / (p * np.sqrt(a)),
                           "depth_proxy": np.sqrt(n * a) / p,
                           "density": n / a,
                           "dd/p90": n**1.5 / (p * np.sqrt(a)) / p90}.items():
                masked = np.where(elig & np.isfinite(sc), sc, -np.inf)
                rank = {bid[i]: r + 1 for r, i in enumerate(np.argsort(-masked))}
                series[f"{city[:2]}:{variant}:{nm}"] = curve(fine_order, rank)

    print(f"  {'series':<34}" + "".join(f"{f'f({k})':>9}" for k in KS) + f"{'  cuts':>8}")
    for nm, f in series.items():
        cells = "".join(f"{(f[k - 1] if k <= len(f) else f[-1]):>9,}" for k in KS)
        closed = int(np.sum(f == np.arange(1, len(f) + 1)))
        print(f"  {nm:<34}{cells}{closed:>8,}")
    print("\n  f(k) = smallest proxy_keep_n retaining the fine top-k;  cuts = #k with f(k)=k")
    print("  (a 'cut' is a closed prefix: the direct-sum decomposition points of the permutation)")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
        for nm, f in series.items():
            k = np.arange(1, len(f) + 1)
            ax.plot(k, f, lw=1.1, label=nm)
        lim = max(len(f) for f in series.values())
        ax.plot([1, lim], [1, lim], "k--", lw=0.8, label="f(k)=k (perfect)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k  (fine metric's top-k)")
        ax.set_ylabel("f(k)  = proxy prefix needed to retain them")
        ax.legend(fontsize=7)
        ax.set_title("Retention curve: prefix maximum of the rank permutation")
        fig.tight_layout()
        fig.savefig(args.png)
        print(f"  wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
