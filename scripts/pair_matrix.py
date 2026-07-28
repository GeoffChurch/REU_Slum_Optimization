"""GW pair-matrix benchmark (Phase 1, unit 1d).

Per (recipient, donor): fit real entropic GW, transplant the donor's linework, snap it to the
recipient's substrate, and score it against a length-matched direct clearance solve. The output
parquet is a retrieval benchmark -- any future featurization or donor material can be scored
against it without re-solving anything.

`load_pools()` selects through the repo's own `Screen` (`density_compactness` = n/P^2 at the
calibrated ABSOLUTE floor) and `RegionBuilder`, rather than the private
`building_count in [60,300] AND k_complexity >= 4` band it used to carry. That band was a separate
population from the one every shipped method is scored on, which made none of this script's
numbers comparable to theirs. See `default_screen` for why depth is reported rather than gated.

It still reads Cape Town from ``~/.cache/reblock/{blocks,buildings}_capetown_full.parquet``. The
census -> shortlist -> provisioned-points chain HAS now run (2026-07-28: 238,484 blocks censused,
9.81M Open Buildings points over 65,364 qualified blocks, see
notes/2026-07-28-osm-census-results.md), but pointing this pilot at it is a separate step -- the
screen over that corpus flags ~19.6k blocks and building them all is ~44 min of Voronoi, so it
needs a sampling decision this script does not yet make.

See docs/superpowers/notes/2026-07-27-gw-pair-matrix-findings.md for the full writeup, and
docs/superpowers/notes/2026-07-23-ot-road-transplant.md for the GW+UOT mechanism this script
drives.

Usage (module form -- puts the repo root on sys.path so `reblock.data.provision`'s
`from scripts.fetch_kblock_fixtures import ...` resolves; see
`scripts/fetch_desire_lines_snapshot.py` for the same convention):
    pixi run python -m scripts.pair_matrix --pairs 20 --timing-only
    pixi run python -m scripts.pair_matrix --pairs 100 --out data/benchmarks/gw_pair_matrix.parquet

`--analyze` re-derives the headline statistics from an already-scored parquet (e.g. the committed
`data/benchmarks/gw_pair_matrix.parquet`) and needs none of the above: no `scratchpad/ot/`, no GW/
OSM/clearance/pool work -- just the parquet plus numpy/pandas/scipy. That guard and the OT imports
are deferred into `_ot()`, called only from the pair-scoring path (`load_pools`, `score_pair`), so
this reproduces the committed headline result from a fresh checkout that never populated
`scratchpad/ot/`:
    pixi run python -m scripts.pair_matrix --analyze --out data/benchmarks/gw_pair_matrix.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from urllib.error import URLError

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from shapely.ops import unary_union

from reblock.budget import building_radii, displacement
from reblock.contracts import Block
from reblock.data.kblock import KblockSource
from reblock.data.osm_extract import (
    PbfDesireLines,
    utm_zone_epsg,
)
from reblock.data.provision import cached_kblock_source
from reblock.data.settlements import exclusion_holdout
from reblock.derivations import access_before
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.desire_lines import DesireLineSource, OSMDesireLines
from reblock.methods.osm_footpaths import interior_desire_lines
from reblock.metric import (
    DENSITY_COMPACTNESS_FLOOR,
    Compactness,
    Density,
    Gate,
    Product,
)
from reblock.permeability import permeability
from reblock.region import IdentityRegionBuilder, RegionBuilder
from reblock.screen.dense_compact import DenseCompactScreen

_OT_DIR = Path("scratchpad/ot")
_ot_ns: SimpleNamespace | None = None


def _ot() -> SimpleNamespace:
    """Lazily import the salvaged 2026-07-23 GW/transplant spike from `scratchpad/ot/`.

    Deferred (not module-level) so `--analyze` -- which reads an already-scored parquet plus
    numpy/pandas/scipy only -- can run in a checkout that lacks `scratchpad/ot/` entirely: it is
    gitignored scratchpad, never repo content, so it does not travel with a fresh checkout. Only
    the pair-scoring path (`load_pools`, `score_pair`) actually needs this; guard + import + raise
    now happen on first call from THAT path, not at import time of this whole module.
    """
    global _ot_ns
    if _ot_ns is not None:
        return _ot_ns
    if not _OT_DIR.is_dir():
        raise SystemExit(
            "scratchpad/ot/ is missing. That directory holds the salvaged 2026-07-23 "
            "GW/transplant spike (ot_gw.py, transplant.py, select_donor.py) this script depends "
            "on -- it is gitignored scratchpad, never repo content, so it does not travel with a "
            "fresh checkout. Rebuild it from "
            "docs/superpowers/notes/2026-07-23-ot-road-transplant.md §1 (entropic GW: "
            "projected-gradient outer loop + log-domain Sinkhorn inner, eps=0.01, tau=1.0) "
            "AND docs/superpowers/notes/2026-07-27-gw-pot-crossvalidation.md, which corrects "
            "that recipe: the outer loop's cost is the GW GRADIENT, 2 * (constC - 2 c1 pi c2^T), "
            "not the undoubled tensor. Omitting the 2 silently doubles both eps and tau (same "
            "argmin under Sinkhorn) and does not reproduce the committed matrix."
        )
    if str(_OT_DIR) not in sys.path:
        sys.path.insert(0, str(_OT_DIR))
    from ot_gw import gw_cost
    from select_donor import signature
    from transplant import _normalized_dist_matrix, fit_transport, gap_snap, transport_lines

    _ot_ns = SimpleNamespace(
        gw_cost=gw_cost, signature=signature, normalized_dist_matrix=_normalized_dist_matrix,
        fit_transport=fit_transport, gap_snap=gap_snap, transport_lines=transport_lines)
    return _ot_ns

CORRIDOR_M = 3.0
DEFAULT_CACHE = Path.home() / ".cache" / "reblock"
# COMPUTE bounds, not a quality judgement -- what a GW fit can afford, kept deliberately separate
# from what is worth reblocking (that is the screen's job, see `default_screen`). GW is quadratic
# in parcel count, and `select_donor.signature`'s fixed subsample (N_SUB=50) cannot sign a block
# with fewer real parcels than that at all.
MIN_BUILDING_COUNT = 60
MAX_BUILDING_COUNT = 300
MIN_PARCELS = 50
CACHE_SHORTLIST_BLOCKS = DEFAULT_CACHE / "blocks_shortlist.parquet"
CACHE_SHORTLIST_BUILDINGS = DEFAULT_CACHE / "buildings_shortlist.parquet"


def default_screen(min_buildings: int = 30) -> DenseCompactScreen:
    """The repo's own screen, `density_compactness` = n/P^2 at its calibrated absolute floor.

    This replaces a hand-rolled `building_count in [60,300] AND k_complexity >= 4` band that this
    script used to define its own pool with. That band was a private population: every number the
    OT arc produced on it -- including the within-recipient slope -- was measured on a different
    set of blocks from the one `clearance`, `arterial` and every other shipped method is scored
    on, so none of the results could be compared across. Selecting through `Screen` is what makes
    them commensurable.

    The gate is the ABSOLUTE calibrated floor (`DENSITY_COMPACTNESS_FLOOR`), not a percentile:
    a percentile re-defines the population every time the corpus changes, and this pilot is meant
    to scale from Cape Town to the ZAF+KEN corpus, where the same percentile is a four-times
    different cut.

    `density_compactness` is also peel-free (`needs_peel=False`), so selection reads the free
    kblock columns and never builds a Block: no Voronoi, no peel, no building points required to
    decide the pool. Depth is deliberately NOT gated on here -- it is the direct measure of the
    access problem and belongs in the OUTPUT as a stratifier, because gating on it would destroy
    the ability to ask whether transplant fidelity depends on it.
    """
    return DenseCompactScreen(
        Product(name="density_compactness", terms=(Density(), Compactness())),
        Gate(kind="absolute", value=DENSITY_COMPACTNESS_FLOOR),
        min_buildings=min_buildings,
    )


def zone_source(epsg: int, *, min_buildings: int = 30) -> KblockSource:
    """A `KblockSource` over the provisioned ZAF+KEN shortlist, restricted to ONE UTM zone.

    The restriction is not optional. `KblockSource.region()` calls `estimate_utm_crs()` on the
    WHOLE blocks frame -- deliberately, so the CRS stays stable under `block_ids` filtering -- so
    pointing it at a two-country shortlist would hand every block a single UTM zone and distort
    area, perimeter and every distance for anything far from that meridian. Filtering by
    `block_ids` does NOT fix it, precisely because of that stability guarantee; the zone subset
    has to be its own parquet, which is what this materializes (once, then cached).

    Splitting by zone costs the cross-zone donor pairs. That is a real loss -- geographic distance
    was measured to be uninformative about GW distance, so a Gauteng donor for a Cape Town
    recipient is not a priori worse -- but it is the honest option until scoring is made
    CRS-aware, and each zone is a genuinely different metro, which makes zone-wise runs a
    REPLICATION rather than merely a smaller sample.
    """
    blocks = CACHE_SHORTLIST_BLOCKS
    buildings = CACHE_SHORTLIST_BUILDINGS
    for path in (blocks, buildings):
        if not path.exists():
            raise SystemExit(
                f"missing {path} -- run `python -m scripts.provision_shortlist` first")
    zone_path = blocks.with_name(f"blocks_shortlist_z{epsg}.parquet")
    if not zone_path.exists():
        frame = gpd.read_parquet(blocks)
        rep = frame.geometry.representative_point()
        keep = np.array([utm_zone_epsg(pt.x, pt.y) == epsg for pt in rep])
        subset = cast(gpd.GeoDataFrame, frame[keep].reset_index(drop=True))
        if subset.empty:
            raise SystemExit(f"no shortlist blocks in UTM zone {epsg}")
        subset.to_parquet(zone_path)
        print(f"  materialized {zone_path.name}: {len(subset):,} blocks", flush=True)
    return KblockSource(zone_path, buildings, region_id=f"shortlist-z{epsg}",
                        min_buildings=min_buildings)


def displacement_fraction(block: Block, roads: gpd.GeoDataFrame) -> float:
    """Expected homes displaced as a fraction of the block's buildings.

    `budget.displacement` takes `(building_points, radii, roads, corridor_m)` and returns a COUNT,
    not a fraction and not a Block -- this mirrors the normalization in `emit.py:92`
    (`pct_displaced`).
    """
    pts = block.building_points
    n = len(pts)
    if n == 0:
        return 0.0
    radii = building_radii(pts, CORRIDOR_M)
    return float(displacement(pts, radii, roads, CORRIDOR_M) / n)


def load_pools(
    city: str = "capetown", *, min_buildings: int = 30,
    screen: DenseCompactScreen | None = None,
    region_builder: RegionBuilder | None = None,
    source: KblockSource | None = None,
) -> tuple[list[Block], gpd.GeoDataFrame, dict[str, np.ndarray]]:
    """Real Cape Town blocks WITH building points, so `KblockSource` can build real `Block`s with
    Voronoi parcels (required by `gap_snap` and `permeability`).

    The pool is chosen by `screen` (default `default_screen()`, the repo's `density_compactness`)
    intersected with the MIN/MAX_BUILDING_COUNT compute bounds, then restricted to blocks whose
    real building-point join yields >= `MIN_PARCELS` parcels -- the stored `building_count` column
    is only a proxy for that join, and `select_donor.signature`'s fixed subsample needs a floor.

    `region_builder` (default `IdentityRegionBuilder`) runs over the screen's output as singleton
    seed groups. At singleton granularity identity is a no-op, which is exactly the point: it puts
    this experiment on the same Source -> Screen -> RegionBuilder path every shipped method uses,
    so swapping in an accreting builder later -- which Phase 3's street-form donor material
    REQUIRES, since a single block has no internal streets -- is a substitution here rather than a
    rewrite. A builder that returns non-singleton groups needs `region.region_block` to fuse each
    group before scoring; that path is deliberately not built until something needs it.

    Returns `(blocks, blocks_gdf, signatures)`:
      - `blocks`: the materialized pool as real `Block`s (Voronoi parcels via `KblockSource`).
      - `blocks_gdf`: block boundary geometry in the same order/index as `blocks` -- the
        GeoDataFrame `exclusion_holdout` operates over.
      - `signatures`: block_id -> GW-consistent shape signature (parcel-centroid eigen-spectrum,
        `select_donor.signature`), precomputed once for the whole pool. This is a cheap PROXY for
        the real (expensive) GW distance, used only to stratify candidate donors by similarity
        before paying for a real GW fit -- never written to the output parquet in place of
        `real_gw_dist`.
    """
    screen = screen or default_screen(min_buildings)
    region_builder = region_builder or IdentityRegionBuilder()
    src_all = source or cached_kblock_source(city, min_buildings=min_buildings)

    flagged = screen.select(src_all)
    raw = pd.read_parquet(
        src_all.blocks_path, columns=["block_id", "building_count", "geometry"])
    raw["block_id"] = raw["block_id"].astype(str)
    counts = dict(zip(raw["block_id"], raw["building_count"], strict=True))
    affordable = sorted(
        b for b in flagged
        if MIN_BUILDING_COUNT <= float(counts.get(b, 0)) <= MAX_BUILDING_COUNT
    )
    print(f"  screen flagged {len(flagged):,}; {len(affordable):,} within the "
          f"[{MIN_BUILDING_COUNT}, {MAX_BUILDING_COUNT}] compute band", flush=True)

    groups = region_builder.build(
        cast(gpd.GeoDataFrame, raw[raw["block_id"].isin(affordable)].reset_index(drop=True)),
        [[b] for b in affordable],
    )
    ids = sorted({b for group in groups for b in group})

    src = (KblockSource(src_all.blocks_path, src_all.buildings_path,
                        region_id=src_all.region_id, min_buildings=min_buildings, block_ids=ids)
           if source is not None
           else cached_kblock_source(city, block_ids=ids, min_buildings=min_buildings))
    blocks = [b for b in src.region().blocks if len(b.parcels) >= MIN_PARCELS]
    blocks.sort(key=lambda b: b.block_id)
    if not blocks:
        raise SystemExit(f"load_pools: no screened {city} blocks survived construction")

    blocks_gdf = gpd.GeoDataFrame(
        {"block_id": [b.block_id for b in blocks]},
        geometry=[b.boundary for b in blocks],
        crs=blocks[0].crs,
    )

    signatures: dict[str, np.ndarray] = {}
    for b in blocks:
        xy = np.c_[
            b.parcels.geometry.centroid.x.to_numpy(), b.parcels.geometry.centroid.y.to_numpy()
        ]
        signatures[b.block_id] = _ot().signature(xy)
    return blocks, blocks_gdf, signatures


def _select_recipient_indices(blocks: list[Block], n: int) -> list[int]:
    """`n` pool indices spanning the parcel-count range: sort by parcel count, then take `n`
    evenly-spaced ranks (min, ..., max) -- not a random sample, so the matrix deliberately
    includes the pool's smallest and largest blocks rather than leaving that to chance."""
    order = sorted(range(len(blocks)), key=lambda i: len(blocks[i].parcels))
    if n >= len(order):
        return order
    return [order[int(round(k))] for k in np.linspace(0, len(order) - 1, n)]


def _select_donor_candidates(
    recipient: Block,
    eligible: list[int],
    blocks: list[Block],
    signatures: dict[str, np.ndarray],
    n_candidates: int,
) -> list[int]:
    """Stratify `eligible` donor indices by proxy (signature) distance to `recipient`, then take
    `n_candidates` evenly-spaced ranks -- near, mid, and far in feature space. Real GW distance is
    too expensive to compute for every eligible donor just to pick a spread (that is the whole
    point of the pilot), so this cheap proxy (correlated with, but not identical to, the real GW
    cost -- see `real_gw_dist` vs `feature_dist` in the output) stands in for it at selection time.
    """
    r_sig = signatures[recipient.block_id]
    ranked = sorted(
        eligible, key=lambda j: float(np.linalg.norm(signatures[blocks[j].block_id] - r_sig))
    )
    if n_candidates >= len(ranked):
        return ranked
    return [ranked[int(round(k))] for k in np.linspace(0, len(ranked) - 1, n_candidates)]


def rank1_distance_scaling(
    signatures: dict[str, np.ndarray], sizes: list[int], *, n_trials: int = 200, seed: int = 0
) -> pd.DataFrame:
    """How the nearest-donor (rank-1) proxy-signature distance shrinks as the candidate donor pool
    grows, measured (not assumed from a theoretical N^(-1/d)) by resampling: for each pool size
    `N` in `sizes`, draw `n_trials` random (held-out recipient, N-block donor pool) splits from
    the qualified-pool signatures and record the nearest neighbour's distance. This is entirely a
    function of the cheap signature proxy (already computed by `load_pools` for donor-candidate
    stratification) -- real GW distance is too expensive to pay 1000x per trial x per size just
    to fit a scaling exponent, and the whole point of a PROXY is that its geometry (which is what
    a power-law retrieval-scaling exponent measures) is what donor selection already relies on.
    Returns long-form (pool_size, trial, rank1_dist) rows; fit `log(rank1_dist) ~ log(pool_size)`
    over the per-size median for the exponent.
    """
    ids = list(signatures)
    sig_mat = np.stack([signatures[i] for i in ids])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for size in sizes:
        capped = min(size, len(ids) - 1)
        for trial in range(n_trials):
            recipient_pos = int(rng.integers(len(ids)))
            others = np.delete(np.arange(len(ids)), recipient_pos)
            pool_pos = rng.choice(others, size=capped, replace=False)
            d = np.linalg.norm(sig_mat[pool_pos] - sig_mat[recipient_pos], axis=1)
            rows.append({"pool_size": size, "trial": trial, "rank1_dist": float(d.min())})
    return pd.DataFrame(rows)


# --- Clustering-aware fidelity-vs-distance analysis -------------------------------------------
#
# The 100 scored rows are 20 recipients x ~5 donors each -- clustered, not independent. A naive
# Pearson correlation of `real_gw_dist` against `perm_gap` pooled over all 100 rows lets a
# between-recipient trend cancel or reverse a within-recipient one (Simpson's paradox): recipients
# differ systematically in both their achievable GW-distance range and their baseline `perm_gap`
# level, and pooling conflates "does a closer donor transplant better FOR A GIVEN RECIPIENT" with
# "do recipients whose donors happen to sit at larger GW distances also happen to have larger
# perm_gap for other reasons." The functions below decompose that pooled number rather than
# reporting it alone -- see docs/superpowers/notes/2026-07-27-gw-pair-matrix-findings.md for the
# full writeup and the numbers this analysis actually produced on the committed parquet.


def _demean_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each observation's own group mean -- the within-group ("fixed effects") transform
    that removes all between-group variance, isolating the within-cluster relationship."""
    means = pd.Series(values).groupby(groups).transform("mean").to_numpy()
    return np.asarray(values, dtype=np.float64) - means


def icc_one_way(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects intraclass correlation (Fisher's unbalanced-design ICC(1)): the
    fraction of `values`' total variance sitting BETWEEN groups rather than within them. A large
    ICC on `perm_gap` grouped by `recipient` is precisely the condition under which pooling rows
    across recipients is unsafe (see the module note above)."""
    d = pd.DataFrame({"v": np.asarray(values, dtype=np.float64), "g": groups})
    grand_mean = float(d["v"].mean())
    counts = d.groupby("g")["v"].count()
    means = d.groupby("g")["v"].mean()
    k, n = len(counts), len(d)
    ssb = float((counts * (means - grand_mean) ** 2).sum())
    ssw = float(d.groupby("g")["v"].apply(lambda s: float(((s - s.mean()) ** 2).sum())).sum())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - float((counts**2).sum()) / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom else 0.0


def variance_explained_by_recipient(values: np.ndarray, groups: np.ndarray) -> float:
    """SSB / SST for a one-way ANOVA of `values` on `groups` -- equivalently, the R-squared of
    regressing `values` on recipient dummy variables. Easy to conflate with `icc_one_way` under
    the informal label ICC (both describe how much of the variance is between-group), but they
    answer distinct questions: this is the RAW sample sum-of-squares ratio, while `icc_one_way`
    corrects it for the within-group noise that inflates between-group dispersion even under
    pure noise, for a finite number of small, unequal-sized groups (this pool: 20 recipients,
    mostly n_i=5). This statistic is always >= the corrected ICC(1) for that reason; reported
    alongside it, not instead of it, since both are legitimate descriptions and either may be
    the number a given source calls ICC."""
    d = pd.DataFrame({"v": np.asarray(values, dtype=np.float64), "g": groups})
    grand_mean = float(d["v"].mean())
    ssb = float((d.groupby("g")["v"].transform("mean") - grand_mean).pow(2).sum())
    sst = float(((d["v"] - grand_mean) ** 2).sum())
    return ssb / sst if sst else 0.0


def within_recipient_regression(df: pd.DataFrame) -> dict[str, float]:
    """Fixed-effects (within-recipient / demeaned) OLS slope of `perm_gap` on `real_gw_dist` --
    the correct estimator when recipients, not donors, are the independent sampling unit. Returns
    `beta`, its standard error, t-statistic, degrees of freedom (`N - n_recipients - 1`), and a
    two-sided p-value from the t distribution."""
    groups = df["recipient"].to_numpy()
    x = df["real_gw_dist"].to_numpy(dtype=np.float64)
    y = df["perm_gap"].to_numpy(dtype=np.float64)
    dx, dy = _demean_by_group(x, groups), _demean_by_group(y, groups)
    n, k = len(df), int(df["recipient"].nunique())
    dof = n - k - 1
    sxx = float(np.sum(dx * dx))
    beta = float(np.sum(dx * dy) / sxx)
    resid = dy - beta * dx
    sigma2 = float(np.sum(resid**2) / dof) if dof > 0 else float("nan")
    se = float(np.sqrt(sigma2 / sxx)) if sxx > 0 else float("nan")
    t_stat = beta / se if se else float("nan")
    p_value = float(2 * stats.t.sf(abs(t_stat), dof)) if dof > 0 else float("nan")
    return {"beta": beta, "se": se, "t": t_stat, "dof": float(dof), "p": p_value}


def within_recipient_permutation_test(
    df: pd.DataFrame, *, n_perm: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """Cluster-preserving permutation test for the within-recipient slope: shuffle `real_gw_dist`
    WITHIN each recipient's own rows only (never across recipients -- each recipient's achievable
    GW-distance values and `perm_gap` values both stay fixed; only which donor's distance pairs
    with which donor's perm_gap, within that recipient, is scrambled), recompute the same
    fixed-effects slope, and report the two-sided fraction of permutations at least as extreme as
    the observed slope. A within-group permutation leaves that group's OWN mean (and hence
    `sum(dx**2)` over the whole sample) unchanged, so this reduces to reshuffling the
    already-demeaned `dx` within each recipient's row block. Returns (observed_beta, p_value)."""
    groups = df["recipient"].to_numpy()
    x = df["real_gw_dist"].to_numpy(dtype=np.float64)
    y = df["perm_gap"].to_numpy(dtype=np.float64)
    dx, dy = _demean_by_group(x, groups), _demean_by_group(y, groups)
    sxx = float(np.sum(dx * dx))
    observed = float(np.sum(dx * dy) / sxx)
    idx_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    dx_work = dx.copy()
    null = np.empty(n_perm, dtype=np.float64)
    for p in range(n_perm):
        for idxs in idx_by_group:
            dx_work[idxs] = rng.permutation(dx[idxs])
        null[p] = np.sum(dx_work * dy) / sxx
    p_value = float(np.mean(np.abs(null) >= abs(observed)))
    return observed, p_value


def recipient_level_correlation(df: pd.DataFrame) -> tuple[float, int]:
    """Pearson correlation of PER-RECIPIENT mean `real_gw_dist` vs mean `perm_gap` (one point per
    recipient). This is the between-recipient trend that, pooled naively together with the
    within-recipient rows, can cancel or reverse the within-recipient sign -- report it alongside,
    never in place of, the within-recipient estimate."""
    agg = df.groupby("recipient")[["real_gw_dist", "perm_gap"]].mean()
    return float(agg["real_gw_dist"].corr(agg["perm_gap"])), int(len(agg))


def range_restriction_summary(df: pd.DataFrame) -> dict[str, float]:
    """How restricted this sample's `real_gw_dist` range is relative to the `feature_dist` proxy
    used to STRATIFY donor selection -- range restriction attenuates a detectable correlation's
    MAGNITUDE, not whether an effect exists. `feature_dist` was deliberately stratified near/mid/
    far per recipient; if it tracked `real_gw_dist` perfectly, stratifying on one would stratify
    the other by the same ratio. A max/min ratio much narrower than feature_dist's own ratio, on
    the identical 100 pairs, shows the stratification did not transfer proportionally -- i.e. the
    achieved `real_gw_dist` range likely undersamples the full range achievable in the pool."""
    gw, fd = df["real_gw_dist"], df["feature_dist"]
    return {
        "real_gw_dist_min": float(gw.min()),
        "real_gw_dist_max": float(gw.max()),
        "real_gw_dist_mean": float(gw.mean()),
        "real_gw_dist_sd": float(gw.std()),
        "real_gw_dist_max_over_min": float(gw.max() / gw.min()),
        "feature_dist_min": float(fd.min()),
        "feature_dist_max": float(fd.max()),
        "feature_dist_max_over_min": float(fd.max() / fd.min()),
        "gw_feature_corr": float(gw.corr(fd)),
    }


def donor_bootstrap(
    df: pd.DataFrame, *, n_boot: int = 4000, seed: int = 0
) -> dict[str, float]:
    """Cluster bootstrap over the DONOR dimension: resample each recipient's donors with
    replacement, refit the within-recipient slope, and report the distribution.

    This exists because a single-draw p-value from this design was shown to be meaningless. Two
    runs over the same pool with the SAME 20 recipients, differing only in which donors were
    selected, gave beta=-9.59 (p=0.014) and beta=-3.97 (p=0.31) -- and the 10 pairs they shared
    scored identically, so it was purely the draw. Re-scoring at 25 donors per recipient put the
    5-donor design's beta interval at [-11.2, +5.4] with the published -9.58 at its 5th
    percentile. See notes/2026-07-28-beta-was-a-lucky-draw.md.

    So: report an interval, never a point. `beta_negative_frac` is the honest headline -- the
    share of donor resamples in which the effect points the claimed way at all.
    """
    rng = np.random.default_rng(seed)
    groups = [g.reset_index(drop=True) for _, g in df.groupby("recipient")]
    betas: list[float] = []
    for _ in range(n_boot):
        draw = pd.concat(
            [g.iloc[rng.integers(0, len(g), len(g))] for g in groups], ignore_index=True)
        sub = draw[["recipient", "real_gw_dist", "perm_gap"]]
        if sub["recipient"].nunique() < 2:
            continue
        betas.append(within_recipient_regression(sub)["beta"])
    b = np.asarray(betas, dtype=np.float64)
    return {
        "beta_median": float(np.median(b)),
        "beta_lo95": float(np.percentile(b, 2.5)),
        "beta_hi95": float(np.percentile(b, 97.5)),
        "beta_sd": float(b.std()),
        "beta_negative_frac": float((b < 0).mean()),
        "n_boot": float(len(b)),
    }


def analyze_fidelity_vs_distance(
    df: pd.DataFrame, *, n_perm: int = 5000, seed: int = 0
) -> dict[str, object]:
    """The full clustering-aware analysis of an already-scored pair-matrix parquet: the naive
    pooled correlation, the recipient-clustering ICC that explains why pooling it is unsafe, the
    within-recipient fixed-effects slope with its parametric and permutation significance, the
    recipient-level aggregate correlation (the pooled number's cancelling counterpart), a
    robustness check dropping the zero-road-length rows, a leave-one-recipient-out jackknife of
    the within-recipient slope, and the range-restriction summary. Never re-fits any GW pair --
    purely a function of an already-scored matrix's columns."""
    within = within_recipient_regression(df)
    _observed, perm_p = within_recipient_permutation_test(df, n_perm=n_perm, seed=seed)
    bootstrap = donor_bootstrap(df, seed=seed)
    recipient_r, n_recipients = recipient_level_correlation(df)
    nonzero = df[df["road_len_m"] > 0]
    jackknife = [
        within_recipient_regression(df[df["recipient"] != rid])["beta"]
        for rid in df["recipient"].unique()
    ]
    return {
        "n": len(df),
        "n_recipients": n_recipients,
        "pooled_pearson_r": float(df["real_gw_dist"].corr(df["perm_gap"])),
        "icc_perm_gap_by_recipient": icc_one_way(
            df["perm_gap"].to_numpy(dtype=np.float64), df["recipient"].to_numpy()
        ),
        "variance_explained_by_recipient": variance_explained_by_recipient(
            df["perm_gap"].to_numpy(dtype=np.float64), df["recipient"].to_numpy()
        ),
        "within_recipient": within,
        "within_recipient_permutation_p": perm_p,
        "donor_bootstrap": bootstrap,
        "recipient_level_r": recipient_r,
        "within_recipient_excl_zero_length": within_recipient_regression(nonzero),
        "jackknife_beta_min": float(min(jackknife)),
        "jackknife_beta_max": float(max(jackknife)),
        "range_restriction": range_restriction_summary(df),
    }


def _print_analysis(result: dict[str, object]) -> None:
    within = result["within_recipient"]
    within_nz = result["within_recipient_excl_zero_length"]
    restriction = result["range_restriction"]
    assert isinstance(within, dict)
    assert isinstance(within_nz, dict)
    assert isinstance(restriction, dict)
    print(f"n = {result['n']} rows, {result['n_recipients']} recipients")
    print(
        f"pooled Pearson r(real_gw_dist, perm_gap) = {result['pooled_pearson_r']:.4f}  "
        "<-- artifact, see below"
    )
    print(f"ICC(1) (perm_gap by recipient, unbalanced-corrected) = "
          f"{result['icc_perm_gap_by_recipient']:.4f}")
    print(f"  R^2 / eta^2 (raw SSB/SST, uncorrected)             = "
          f"{result['variance_explained_by_recipient']:.4f}")
    print(
        f"within-recipient beta (perm_gap ~ real_gw_dist)  = {within['beta']:.4f}  "
        f"SE={within['se']:.4f}  t={within['t']:.4f}  dof={within['dof']:.0f}"
    )
    print(f"  p (t-distribution)      = {within['p']:.4f}")
    print(f"  p (cluster permutation) = {result['within_recipient_permutation_p']:.4f}")
    bs = cast(dict[str, float], result["donor_bootstrap"])
    print(
        f"  DONOR BOOTSTRAP ({bs['n_boot']:.0f} resamples) -- read this, not the point estimate:\n"
        f"    beta 95% interval [{bs['beta_lo95']:+.3f}, {bs['beta_hi95']:+.3f}]  "
        f"median {bs['beta_median']:+.3f}  sd {bs['beta_sd']:.3f}\n"
        f"    negative in {bs['beta_negative_frac']:.1%} of donor resamples"
    )
    print(
        f"recipient-level aggregate r (n={result['n_recipients']}) = "
        f"{result['recipient_level_r']:.4f}  <-- the cancelling counterpart"
    )
    print(
        f"within-recipient beta, excl. zero-length rows = {within_nz['beta']:.4f}  "
        f"p={within_nz['p']:.4f}"
    )
    print(
        "jackknife beta range (leave-one-recipient-out) = "
        f"[{result['jackknife_beta_min']:.4f}, {result['jackknife_beta_max']:.4f}]"
    )
    print("range restriction:")
    print(
        f"  real_gw_dist: min={restriction['real_gw_dist_min']:.4f} "
        f"max={restriction['real_gw_dist_max']:.4f} mean={restriction['real_gw_dist_mean']:.4f} "
        f"sd={restriction['real_gw_dist_sd']:.4f}"
    )
    print(f"    max/min = {restriction['real_gw_dist_max_over_min']:.2f}x")
    print(
        f"  feature_dist: min={restriction['feature_dist_min']:.4f} "
        f"max={restriction['feature_dist_max']:.4f}"
    )
    print(f"    max/min = {restriction['feature_dist_max_over_min']:.2f}x")
    print(f"  corr(real_gw_dist, feature_dist) = {restriction['gw_feature_corr']:.4f}")


def _donor_bbox_wgs84(donor: Block) -> tuple[float, float, float, float]:
    b = gpd.GeoSeries([donor.boundary], crs=donor.crs).to_crs(4326).total_bounds
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


PBF_BY_ISO = {"ZAF": "south-africa-latest.osm.pbf", "KEN": "kenya-latest.osm.pbf"}


def iso_of(blocks: list[Block]) -> str:
    """The country a pool belongs to, from its kblock ids (`ZAF.9.3.1_1_44882`, `KEN.1.1_1_100`).

    Load-bearing, and learned the hard way: a PBF covers EXACTLY its own extract, so pointing a
    Kenyan pool at the South Africa extract does not error -- every donor simply comes back with
    no interior footpaths. The first Nairobi run reported `empty_interior: 90` and zero pairs,
    which reads as "these blocks have no footpaths" and is flatly contradicted by the census (56
    of them carry >=100 m each). Deriving the extract from the data removes the chance to pick
    the wrong one by hand.
    """
    isos = {str(b.block_id).split(".", 1)[0] for b in blocks}
    if len(isos) != 1:
        raise SystemExit(f"pool spans multiple countries {sorted(isos)}; one PBF cannot cover it")
    iso = isos.pop()
    if iso not in PBF_BY_ISO:
        raise SystemExit(
            f"no Geofabrik extract configured for {iso!r}; "
            f"known: {sorted(PBF_BY_ISO)}")
    return iso


def desire_source(kind: str, iso: str = "ZAF") -> DesireLineSource:
    """`pbf` (default) reads the local Geofabrik extract FOR `iso`; `overpass` hits the live API.

    Not a fallback pair -- two live options with disjoint operating ranges. A PBF covers exactly
    its own extract and nothing outside it; Overpass covers any bbox on earth but is a shared
    third-party service that, measured here, failed 214 of 241 donor fetches in one run.
    """
    if kind == "overpass":
        return OSMDesireLines()
    pbf = DEFAULT_CACHE / "osm_pbf" / PBF_BY_ISO[iso]
    if not pbf.exists():
        raise SystemExit(
            f"missing {pbf}\ndownload it from https://download.geofabrik.de/, or pass "
            f"--desire-source overpass to use the live API instead.")
    return PbfDesireLines(pbf_path=pbf)


def fetch_donor_lines(
    source: DesireLineSource, donor: Block, *, max_tries: int = 4, base_backoff_s: float = 2.0
) -> tuple[str, gpd.GeoDataFrame | None]:
    """Donor material: the donor block's real interior OSM footpaths (`donor_type =
    "osm_footpaths"`), mirroring `OsmFootpathsReblocker.propose`.

    `source` is any `DesireLineSource`. The default is now `PbfDesireLines` over a local Geofabrik
    extract, because Overpass could not carry this: a 100-pair run against it returned 27 usable
    pairs and 214 `fetch_failed`, spending 4,440 s -- 100% of wall clock -- against 15 s of actual
    GW, transplant, clearance and permeability work. The PBF reads once into memory and every
    donor after that is a bbox window, so the same run is one ~40 s read plus no network at all.

    The retry/backoff below is dead weight against a PBF and deliberately kept: `OSMDesireLines`
    remains a legitimate choice (Overpass covers any bbox on earth; a PBF covers its own extract),
    so this stays useful whenever the source IS a network one. A donor whose lines can't be
    fetched after `max_tries` is reported as `"fetch_failed"` -- the caller must skip it and count
    the skip, never silently drop it from the totals. `"empty_interior"` means the fetch succeeded
    but the donor has no interior footpath material once perimeter-retracing streets are
    subtracted -- also a skip, not a zero-length row.
    """
    bbox = _donor_bbox_wgs84(donor)
    lines = None
    for attempt in range(max_tries):
        try:
            lines = source.desire_lines(bbox, donor.crs)
            break
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            if attempt == max_tries - 1:
                return "fetch_failed", None
            wait = base_backoff_s * (2**attempt)
            print(f"    retry {donor.block_id} OSM fetch in {wait:.0f}s ({exc!r})")
            time.sleep(wait)
    assert lines is not None
    streets = unary_union(list(donor.streets.geometry))
    interior = interior_desire_lines(lines, donor.boundary, streets, donor.crs)
    if interior.empty:
        return "empty_interior", None
    return "ok", interior


def score_pair(
    recipient: Block, donor: Block, donor_lines: gpd.GeoDataFrame, timings: dict[str, float]
) -> dict[str, object]:
    """One matrix row. `recipient`/`donor` are Blocks; `donor_lines` is the donor's material."""
    ot = _ot()
    r_xy = np.c_[recipient.parcels.geometry.centroid.x, recipient.parcels.geometry.centroid.y]
    d_xy = np.c_[donor.parcels.geometry.centroid.x, donor.parcels.geometry.centroid.y]

    t = time.time()
    result = ot.fit_transport(d_xy, r_xy, eps=0.01, tau=1.0)
    timings["gw"] += time.time() - t
    dist = ot.gw_cost(result.pi, ot.normalized_dist_matrix(d_xy), ot.normalized_dist_matrix(r_xy))

    t = time.time()
    warped = ot.transport_lines(donor_lines, result, out_crs=recipient.crs)
    moved = ot.gap_snap(warped, recipient)
    timings["transplant"] += time.time() - t

    t = time.time()
    road_len = float(moved.geometry.length.sum())
    direct = ClearanceReblocker().propose(recipient).roads
    # Length-match the baseline by truncating to a prefix of comparable total length.
    cum = direct.geometry.length.cumsum()
    direct = direct[cum <= road_len] if road_len > 0 else direct.iloc[:0]
    timings["clearance"] += time.time() - t

    t = time.time()
    perm_prop = permeability(recipient, moved)
    perm_direct = permeability(recipient, direct)
    timings["permeability"] += time.time() - t

    return {
        "recipient": recipient.block_id,
        "donor": donor.block_id,
        "donor_type": "osm_footpaths",
        # Depth is REPORTED, never gated on. It is the direct measure of the access problem
        # reblocking exists to fix, and the screen (`density_compactness` = n/P^2) does not
        # capture it -- measured on Cape Town, only 29% of the screened pool reaches k>=4, where
        # the old hand-rolled band required it of everything. Carrying it as a column is what
        # lets the analysis ask whether transplant fidelity depends on depth; a gate would have
        # made that question unanswerable from the matrix.
        "recipient_depth": float(access_before(recipient).max()),
        "donor_depth": float(access_before(donor).max()),
        "real_gw_dist": float(dist),
        "feature_dist": float(np.linalg.norm(ot.signature(d_xy) - ot.signature(r_xy))),
        "perm_gap": float(perm_prop - perm_direct),
        "perm_proposal": float(perm_prop),
        "perm_direct": float(perm_direct),
        "displacement_proposal": displacement_fraction(recipient, moved),
        "displacement_direct": displacement_fraction(recipient, direct),
        "road_len_m": road_len,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    # 500 pairs / 25 donors per recipient, not the original 100 / 5. At 5 donors the design's
    # standard error (3.81) exceeded the effect it was measuring, so which donors got drawn
    # decided the result -- two runs over the same pool with the same recipients gave -9.59
    # (p=0.014) and -3.97 (p=0.31). 25 donors halves the SE to 1.60. The old numbers were sized
    # for when every donor meant an Overpass round trip; off the local PBF this is ~8 minutes.
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--exclusion-radius-m", type=float, default=2000.0)
    ap.add_argument("--donors-per-recipient", type=int, default=25)
    ap.add_argument(
        "--candidate-multiplier",
        type=int,
        default=3,
        help="try up to this many x donors-per-recipient candidates per recipient, backfilling "
        "around OSM fetch/empty-interior skips",
    )
    ap.add_argument("--out", type=Path, default=Path("data/benchmarks/gw_pair_matrix.parquet"))
    ap.add_argument(
        "--utm-zone", type=int, default=None,
        help="run over the provisioned ZAF+KEN shortlist restricted to this UTM EPSG (e.g. 32735 "
             "= Gauteng, 32734 = Cape Town) instead of the cached Cape Town city parquets. One "
             "zone at a time because KblockSource assigns a single estimate_utm_crs per parquet")
    ap.add_argument(
        "--desire-source", choices=("pbf", "overpass"), default="pbf",
        help="where donor footpaths come from. pbf (default) reads the local Geofabrik extract "
             "once into memory and windows it per donor -- no network; overpass hits the live API")
    ap.add_argument(
        "--rank1-scaling",
        type=str,
        default=None,
        help="comma-separated pool sizes (e.g. '10,30,100,300,1000'); if set, run ONLY the "
        "rank-1-distance pool-size scaling analysis (cheap, signature-proxy-only) and exit "
        "without touching GW/OSM/clearance at all",
    )
    ap.add_argument(
        "--analyze",
        action="store_true",
        help="load --out and run the clustering-aware fidelity-vs-distance analysis (pooled "
        "correlation, ICC, within-recipient fixed-effects slope + permutation test, "
        "recipient-level aggregate, jackknife, range restriction) and exit; reads the parquet "
        "only, no GW/OSM/clearance/pool work at all",
    )
    args = ap.parse_args()

    if args.analyze:
        if not args.out.exists():
            raise SystemExit(f"--analyze: {args.out} does not exist")
        _print_analysis(analyze_fidelity_vs_distance(pd.read_parquet(args.out)))
        return

    print("loading pools...")
    t_load = time.time()
    blocks, blocks_gdf, signatures = load_pools(
        source=zone_source(args.utm_zone) if args.utm_zone else None)
    where = f"UTM {args.utm_zone}" if args.utm_zone else "Cape Town"
    print(f"  {len(blocks)} screened {where} blocks in {time.time() - t_load:.1f}s")

    if args.rank1_scaling is not None:
        sizes = [int(s) for s in args.rank1_scaling.split(",")]
        df = rank1_distance_scaling(signatures, sizes)
        medians = df.groupby("pool_size")["rank1_dist"].median()
        print(medians)
        log_n, log_d = np.log(medians.index.to_numpy()), np.log(medians.to_numpy())
        slope, intercept = np.polyfit(log_n, log_d, 1)
        print(f"fitted exponent (slope of log(rank1_dist) ~ log(pool_size)): {slope:.4f}")
        return

    n_recipients = max(1, -(-args.pairs // args.donors_per_recipient))  # ceil
    recipient_idx = _select_recipient_indices(blocks, n_recipients)

    source = desire_source(args.desire_source, iso_of(blocks))
    donor_cache: dict[str, tuple[str, gpd.GeoDataFrame | None]] = {}

    # Resume support: this process has no reliable long-lived background execution in this
    # environment (a prior run_in_background attempt was killed with no trace and no
    # notification), so a run long enough to need the full ~90 s Bash-tool timeout budget must be
    # split into several bounded foreground invocations. Each successful row is checkpointed to
    # `args.out` immediately (below) so a kill loses at most the row in flight, and re-running the
    # same command picks up where it left off rather than re-scoring (and re-hitting Overpass for)
    # pairs already on disk. Never applies to --timing-only, which is a throwaway measurement.
    rows: list[dict[str, object]] = []
    done_pairs: set[tuple[str, str]] = set()
    if not args.timing_only and args.out.exists():
        existing = pd.read_parquet(args.out)
        rows = existing.to_dict("records")
        done_pairs = {(str(r["recipient"]), str(r["donor"])) for r in rows}
        print(f"resuming from {args.out}: {len(rows)} rows already scored")

    # Skip counts get the SAME checkpoint-every-update treatment as rows, for the same reason:
    # a chunk's skip tally is otherwise only ever printed (never persisted), so a kill loses it
    # completely (this is exactly how the first chunk's skip tally was lost during Task 9's
    # original run -- see the findings note). The sidecar lives next to the parquet, keyed by its
    # stem, and accumulates across resumed invocations exactly like the parquet's rows do.
    skip_path = args.out.parent / f"{args.out.stem}.skips.json"
    skip_counts: Counter[str] = Counter()
    if not args.timing_only and skip_path.exists():
        skip_counts.update(json.loads(skip_path.read_text()))
    skips_at_start = Counter(skip_counts)

    def _bump_skip(reason: str) -> None:
        skip_counts[reason] += 1
        if not args.timing_only:
            skip_path.parent.mkdir(parents=True, exist_ok=True)
            skip_path.write_text(json.dumps(dict(skip_counts)))

    timings = {"osm_fetch": 0.0, "gw": 0.0, "transplant": 0.0, "clearance": 0.0,
               "permeability": 0.0}
    n_new = 0
    t0 = time.time()

    for i in recipient_idx:
        if len(rows) >= args.pairs:
            break
        recipient = blocks[i]
        eligible = exclusion_holdout(blocks_gdf, i, radius_m=args.exclusion_radius_m)
        n_want = min(args.donors_per_recipient, args.pairs - len(rows))
        candidates = _select_donor_candidates(
            recipient, eligible, blocks, signatures, n_want * args.candidate_multiplier
        )
        got = 0
        for j in candidates:
            if got >= n_want or len(rows) >= args.pairs:
                break
            donor = blocks[j]
            if (recipient.block_id, donor.block_id) in done_pairs:
                got += 1  # already scored in a prior (resumed) invocation -- count, don't redo
                continue
            if donor.block_id not in donor_cache:
                fetch_t0 = time.time()
                status, lines = fetch_donor_lines(source, donor)
                timings["osm_fetch"] += time.time() - fetch_t0
                donor_cache[donor.block_id] = (status, lines)
                if status != "ok":
                    _bump_skip(status)
                    print(f"    skip donor {donor.block_id}: {status}")
            status, lines = donor_cache[donor.block_id]
            if status != "ok" or lines is None:
                continue
            pair_t0 = time.time()
            try:
                row = score_pair(recipient, donor, lines, timings)
            except Exception as exc:  # noqa: BLE001 -- one bad pair must not sink a long run
                _bump_skip("scoring_error")
                print(f"    skip pair ({recipient.block_id}, {donor.block_id}): {exc!r}")
                continue
            row["wall_clock_s"] = time.time() - pair_t0
            rows.append(row)
            done_pairs.add((recipient.block_id, donor.block_id))
            got += 1
            n_new += 1
            print(
                f"  [{len(rows)}/{args.pairs}] {recipient.block_id} <- {donor.block_id}: "
                f"{row['wall_clock_s']:.1f}s"
            )
            if not args.timing_only:
                # Checkpoint after EVERY row, not just at exit -- a kill loses at most one row.
                args.out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_parquet(args.out)

    elapsed = time.time() - t0
    rate = elapsed / max(n_new, 1)
    print(f"\n{n_new} new pairs in {elapsed:.0f}s -- {rate:.1f}s/pair (this run)")
    for stage, secs in timings.items():
        print(f"  {stage:14s} {secs:7.1f}s  ({secs / max(elapsed, 1e-9) * 100:.0f}%)")
    this_run_skips = Counter(skip_counts)
    this_run_skips.subtract(skips_at_start)
    print(f"skips this run: {dict(+this_run_skips)}")  # unary + drops non-positive entries
    if args.timing_only:
        print("(timing-only: nothing written)")
    else:
        print(f"skips all-time (persisted at {skip_path}): {dict(skip_counts)}")
        print(f"total rows on disk: {len(rows)}")


if __name__ == "__main__":
    main()
