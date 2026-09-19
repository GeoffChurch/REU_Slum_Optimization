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
from reblock.metric import _cols

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
            # `_cols`, not `bl.geometry.length`: the blocks parquet is in a GEOGRAPHIC CRS, so a
            # raw `.length` is in DEGREES while `block_area_m2` is metres^2. Mixing them silently
            # produced perimeters ~1e-5 of the truth and an A/P "hydraulic radius" in the millions
            # of metres. `_cols` reprojects to UTM first, and using it here is also what keeps
            # this measurement identical to what `metric.proxy()` scores in production.
            n_s, a_s, p_s = _cols(bl)
            n = n_s.to_numpy(dtype=float)
            a = a_s.to_numpy(dtype=float)
            p = p_s.to_numpy(dtype=float)
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
        _plot(series, args.png, ratio=False)
        _plot(series, args.png.replace(".png", "_ratio.png"), ratio=True)
    return 0


# Colour per PROXY, shared across panels, so four colours are learned once. Emphasis is by ROLE --
# the matched/shipped proxy is bold -- and deliberately NOT by score: encoding performance as
# opacity would fade out exactly the mismatched and baseline series whose shape is the finding.
_COLOR = {"dd_proxy": "#d1495b", "depth_proxy": "#00798c", "density": "#edae49",
          "dd/p90": "#66a182"}
_MATCHED = {"depth": "depth_proxy", "depth_density": "dd_proxy"}
SHIPPED_N = 1000        # conf/config.yaml: proxy_keep_n


def _plot(series: dict[str, NDArray[np.int64]], path: str, *, ratio: bool) -> None:
    """One panel per (city, fine metric), because curves for DIFFERENT fine metrics are not
    comparable -- they measure retention of different target rankings, and sharing an axis invites
    a false read. Only proxies, which do share a target, are overlaid.

    `ratio=True` plots f(k)/k instead of f(k). That view is worth having because f is a RUNNING
    MAXIMUM, so a flat stretch is ambiguous -- it means either "this proxy ranks these blocks
    well" or "this proxy is already so high that nothing new exceeds it", and only the LEVEL tells
    them apart. Every curve goes flat late in the panel from pure saturation. Dividing by k
    removes that: f(k)/k is the multiplicative overpayment, 1.0 is a perfect proxy, and a curve
    tracking the fine metric trends DOWN rather than flattening.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter
    panels: dict[str, dict[str, NDArray[np.int64]]] = {}
    for key, f in series.items():
        city, variant, proxy = key.split(":")
        panels.setdefault(f"{city}:{variant}", {})[proxy] = f
    ncol = 2
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.4 * nrow), dpi=130, squeeze=False)
    for ax, (name, group) in zip(axes.ravel(), panels.items(), strict=False):
        variant = name.split(":")[1]
        lim = max(len(f) for f in group.values())
        if ratio:
            ax.axhline(1.0, color="#999", ls="--", lw=0.9, zorder=1)
            ax.text(lim, 1.0, " f(k)/k = 1 (perfect) ", color="#999", fontsize=7, va="bottom",
                    ha="right")
        else:
            ax.plot([1, lim], [1, lim], color="#999", ls="--", lw=0.9, zorder=1)
            ax.text(lim, lim, " f(k)=k", color="#999", fontsize=7, va="top", ha="right")
            ax.axhline(SHIPPED_N, color="#444", lw=0.9, ls=":", zorder=1)
            ax.text(1.1, SHIPPED_N, f" proxy_keep_n = {SHIPPED_N:,}", fontsize=7, color="#444",
                    va="bottom")
        for proxy, f in group.items():
            matched = proxy == _MATCHED.get(variant)
            k = np.arange(1, len(f) + 1)
            y = f / k if ratio else f
            ax.plot(k, y, color=_COLOR[proxy], lw=2.0 if matched else 1.0,
                    alpha=1.0 if matched else 0.75, label=proxy, zorder=3 if matched else 2)
            if matched and ratio:
                a = int(np.argmax(f / k))
                ax.annotate(f"worst {f[a] / (a + 1):,.0f}x at k={a + 1:,}",
                            xy=(a + 1, f[a] / (a + 1)), xytext=(6, 6),
                            textcoords="offset points", fontsize=7, color=_COLOR[proxy],
                            fontweight="bold")
            elif matched:
                kmax = int(np.searchsorted(f, SHIPPED_N, side="right"))
                ax.annotate(f"retains top-{kmax:,}", xy=(max(kmax, 1), SHIPPED_N),
                            xytext=(4, -14), textcoords="offset points", fontsize=7,
                            color=_COLOR[proxy], fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Integer minor ticks out to k=30: the log decade 1-10 is where every operating point
        # lives (top-1, top-5, top-15) and a bare log axis makes those indistinguishable.
        ax.xaxis.set_minor_locator(FixedLocator(list(range(1, 31))))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=3.5, color="#555", width=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylabel("f(k)/k = overpayment" if ratio else "f(k) = prefix needed")
        ax.grid(alpha=0.15, lw=0.5)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    head = ("Overpayment: f(k)/k, the scale-free view -- 1.0 is a perfect proxy\n"
            "no saturation artifact here: a flat f(k) can just mean 'already high'"
            if ratio else
            "Retention: smallest proxy prefix that keeps the fine metric's top-k\n"
            "bold = the metric's own matched proxy; panels are not comparable across metrics")
    fig.suptitle(head, fontsize=11)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(path)
    print(f"  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
