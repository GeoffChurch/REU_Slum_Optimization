"""Generate the screen bake-off example: which screen metric actually finds informal settlements?

Every other example grades reblocking METHODS. This one grades the SCREENS that choose which blocks
get reblocked at all -- a stage that had never been validated against ground truth until 2026-08-08,
and where the answer turned out to matter: the long-standing `density_compactness` floor selects
1,644 Cape Town blocks of which only 24.5% are really informal settlement.

Ground truth is the City of Cape Town's own informal-structure survey, provisioned by
`reblock.data.informal` (117,336 dwelling polygons, Feb 2018, 1:200, Edinburgh DataShare
doi:10.7488/ds/2758), clustered into 189 settlement extents.

Outputs, into `examples/screen-bakeoff/`:

    screen_comparison.csv   AUC + precision/recall at each retention, per metric
    precision_recall.png    the statistical view
    city_map.png            the whole metro: settlements, and where the two leading screens disagree
    settlements.png         zoomed panels on the settlements where they disagree most
    README.md               written by hand alongside, not generated

Cape Town only -- see `reblock.data.informal` for the (searched, documented) absence of an
equivalent Nairobi layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pyproj import CRS
from scipy.stats import rankdata

from reblock.data.informal import label_blocks, settlement_extents
from reblock.data.provision import cached_kblock_source
from reblock.metric import DENSITY_COMPACTNESS_FLOOR, DEPTH_DENSITY_PROXY_FLOOR

OUT = Path("examples/screen-bakeoff")
UTM = 32734
MIN_COUNT = 30
RETENTIONS = [0.01, 0.05, 0.10, 0.30]

# The four CHEAP metrics -- every one computable from the free kblock columns, no Voronoi, no peel.
# That is deliberate: `density_compactness`'s historical selling point was that it needs no peel,
# and showing its competitors need none either is half the finding.
METRICS = [
    ("dd_proxy", "depth_density proxy   √(nA)/P · n/A", DEPTH_DENSITY_PROXY_FLOOR),
    ("density", "density   n/A", None),
    ("dens_compact", "density_compactness   n/P²", DENSITY_COMPACTNESS_FLOOR),
    ("depth_proxy", "depth proxy   √(nA)/P", None),
]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    r = rankdata(score)
    n1 = int(label.sum())
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[label.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    ext = settlement_extents(epsg=UTM)
    src = cached_kblock_source("capetown", min_buildings=MIN_COUNT)
    raw = gpd.read_parquet(src.blocks_path, columns=["block_id", "building_count", "geometry"])
    raw["block_id"] = raw["block_id"].astype(str)
    projected = raw.crs is not None and CRS.from_user_input(raw.crs).is_projected
    b = raw if projected else raw.to_crs(UTM)
    # bracket notation throughout: `gdf.area` is geopandas' geometry property, shadowing a column
    b = b.assign(a_m2=b.geometry.area.to_numpy(), p_m=b.geometry.length.to_numpy())
    b = b[(b["building_count"] >= MIN_COUNT) & (b["p_m"] > 0) & (b["a_m2"] > 0)].reset_index(
        drop=True)
    b["density"] = b["building_count"] / b["a_m2"]
    b["compact"] = b["a_m2"] / b["p_m"] ** 2
    b["dens_compact"] = b["density"] * b["compact"]
    b["depth_proxy"] = np.sqrt(b["building_count"] * b["a_m2"]) / b["p_m"]
    b["dd_proxy"] = b["depth_proxy"] * b["density"]
    cover, label = label_blocks(b, ext)
    b["cover"], b["informal"] = cover, label.astype(int)
    return b, ext


def table(b: gpd.GeoDataFrame) -> pd.DataFrame:
    lab = b["informal"].to_numpy()
    rows = []
    for col, name, floor in METRICS:
        s = b[col].to_numpy()
        order = np.argsort(-s)
        row = {"metric": name, "auc": auc(s, lab)}
        for r in RETENTIONS:
            k = max(1, int(len(s) * r))
            row[f"prec@{r:.0%}"] = float(lab[order[:k]].mean())
            row[f"recall@{r:.0%}"] = float(lab[order[:k]].sum() / lab.sum())
        if floor is not None:
            sel = s >= floor
            row["floor"] = floor
            row["floor_n"] = int(sel.sum())
            row["floor_prec"] = float(lab[sel].mean())
            row["floor_recall"] = float(lab[sel].sum() / lab.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pr(b: gpd.GeoDataFrame, path: Path) -> None:
    lab = b["informal"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col, name, floor in METRICS:
        s = b[col].to_numpy()
        order = np.argsort(-s)
        ks = np.unique(np.geomspace(20, len(s), 200).astype(int))
        prec = np.array([lab[order[:k]].mean() for k in ks])
        rec = np.array([lab[order[:k]].sum() / lab.sum() for k in ks])
        lw = 2.6 if col == "dd_proxy" else 1.6
        axes[0].plot(rec, prec, lw=lw, label=name.split("   ")[0])
        axes[1].plot(100 * ks / len(s), prec, lw=lw, label=name.split("   ")[0])
        if floor is not None:
            sel = s >= floor
            axes[0].plot(lab[sel].sum() / lab.sum(), lab[sel].mean(), "o", ms=9, mfc="none",
                         mew=2, color=axes[0].lines[-1].get_color())
    axes[0].set_xlabel("recall — share of real informal blocks selected")
    axes[0].set_ylabel("precision — share of selected that are really informal")
    axes[0].set_title("precision vs recall (rings = each metric's shipped absolute floor)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("retention (% of blocks kept), log scale")
    axes[1].set_ylabel("precision")
    axes[1].set_title("precision vs how much you keep")
    for ax in axes:
        ax.axhline(lab.mean(), ls=":", c="grey", lw=1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].annotate(f"base rate {lab.mean():.1%}", (0.02, lab.mean() + 0.01), fontsize=8,
                     color="grey")
    fig.suptitle("Which screen finds real informal settlements? "
                 f"Cape Town, {len(b):,} blocks, {int(lab.sum()):,} informal by the City's survey")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _selections(b: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (b["dd_proxy"].to_numpy() >= DEPTH_DENSITY_PROXY_FLOOR,
            b["dens_compact"].to_numpy() >= DENSITY_COMPACTNESS_FLOOR)


def plot_city(b: gpd.GeoDataFrame, ext: gpd.GeoDataFrame, path: Path) -> None:
    new, old = _selections(b)
    fig, ax = plt.subplots(figsize=(12, 12))
    ext.plot(ax=ax, color="#ffd27f", edgecolor="#d98c00", lw=0.4, zorder=1)
    b[~(new | old)].plot(ax=ax, color="#f2f2f2", edgecolor="#dddddd", lw=0.1, zorder=2)
    b[new & old].plot(ax=ax, color="#4d4d4d", edgecolor="none", zorder=3)
    b[new & ~old].plot(ax=ax, color="#1a9850", edgecolor="none", zorder=4)
    b[old & ~new].plot(ax=ax, color="#d73027", edgecolor="none", zorder=4)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", mfc="#ffd27f", mec="#d98c00", ms=12,
               label=f"real informal settlement ({len(ext)} extents, City survey)"),
        Line2D([], [], marker="s", ls="", color="#4d4d4d", ms=12, label="selected by BOTH screens"),
        Line2D([], [], marker="s", ls="", color="#1a9850", ms=12,
               label="depth_density_proxy ONLY"),
        Line2D([], [], marker="s", ls="", color="#d73027", ms=12,
               label="density_compactness ONLY"),
        Line2D([], [], marker="s", ls="", mfc="#f2f2f2", mec="#dddddd", ms=12,
               label="not selected")], loc="upper right", fontsize=9, frameon=True)
    ax.set_title("Where the two screens disagree — Cape Town metro\n"
                 f"green = gained by the new default ({int((new & ~old).sum()):,} blocks), "
                 f"red = dropped ({int((old & ~new).sum()):,})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_settlements(b: gpd.GeoDataFrame, ext: gpd.GeoDataFrame, path: Path, n: int = 4) -> None:
    """Zoom on the settlements where the two screens disagree most."""
    new, old = _selections(b)
    disagree = new ^ old
    cent = b.geometry.centroid
    pts = gpd.GeoDataFrame(geometry=list(cent), crs=b.crs)
    joined = gpd.sjoin(pts, ext.reset_index(names="sid"), how="left", predicate="within")
    sid = joined.groupby(joined.index)["sid"].first().reindex(range(len(b))).to_numpy()
    counts = pd.Series(sid[disagree]).value_counts()
    top = [int(s) for s in counts.index[:n] if not np.isnan(s)]
    if not top:
        return
    fig, axes = plt.subplots(1, len(top), figsize=(5.2 * len(top), 5.8))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, top, strict=False):
        geom = ext.geometry.iloc[s]
        # SQUARE window on the settlement's centre, so the panels are directly comparable
        cx0, cy0 = geom.centroid.x, geom.centroid.y
        gx0, gy0, gx1, gy1 = geom.bounds
        half = max(gx1 - gx0, gy1 - gy0) / 2.0 + 150.0
        minx, maxx, miny, maxy = cx0 - half, cx0 + half, cy0 - half, cy0 + half
        sub = b.cx[minx:maxx, miny:maxy]
        m = sub.index.to_numpy()
        sub[~(new[m] | old[m])].plot(ax=ax, color="#f2f2f2", edgecolor="#cccccc", lw=0.2, zorder=1)
        sub[new[m] & old[m]].plot(ax=ax, color="#4d4d4d", edgecolor="white", lw=0.2, zorder=2)
        sub[new[m] & ~old[m]].plot(ax=ax, color="#1a9850", edgecolor="white", lw=0.2, zorder=3)
        sub[old[m] & ~new[m]].plot(ax=ax, color="#d73027", edgecolor="white", lw=0.2, zorder=3)
        # the extent goes ON TOP as an outline: drawn underneath as a fill it is entirely hidden by
        # the blocks, which is what a first version did
        gpd.GeoSeries([geom], crs=ext.crs).plot(ax=ax, facecolor="none", edgecolor="#e08214",
                                                lw=2.2, zorder=4)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(f"{int(ext.n_structures.iloc[s]):,} structures · {2 * half / 1000:.1f} km "
                     f"across\n+{int((new[m] & ~old[m]).sum())} gained, "
                     f"−{int((old[m] & ~new[m]).sum())} dropped", fontsize=10)
    fig.suptitle("The same disagreement, zoomed — the four settlements where the screens differ "
                 "most\ngold OUTLINE = real settlement extent · green = gained by the new "
                 "default · red = dropped · dark grey = selected by both", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.86))     # leave room for the two-line suptitle
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading blocks + settlement extents (downloads ~18 MB once)", flush=True)
    b, ext = load()
    lab = b["informal"].to_numpy()
    print(f"  {len(b):,} blocks, {int(lab.sum()):,} informal ({lab.mean():.2%}), "
          f"{len(ext)} settlement extents", flush=True)

    t = table(b)
    t.to_csv(OUT / "screen_comparison.csv", index=False)
    print("\n" + t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    for name, fn in (("precision_recall.png", plot_pr),
                     ("city_map.png", lambda x, p: plot_city(x, ext, p)),
                     ("settlements.png", lambda x, p: plot_settlements(x, ext, p))):
        print(f"  writing {name}", flush=True)
        fn(b, OUT / name)                                          # type: ignore[operator]
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
