"""GW pair-matrix benchmark (Phase 1, unit 1d).

Per (recipient, donor): fit real entropic GW, transplant the donor's linework, snap it to the
recipient's substrate, and score it against a length-matched direct clearance solve. The output
parquet is a retrieval benchmark -- any future featurization or donor material can be scored
against it without re-solving anything.

The mechanism is `reblock.transplant` at its published operating point
(`reblock.transplant.operating_points`), and the blocks come from `reblock.data.pools`, which
selects through the shipped `Source -> Screen -> RegionBuilder` stages rather than a private band.

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
`data/benchmarks/gw_pair_matrix.parquet`) with no GW, OSM, clearance or pool work at all:
    pixi run python -m scripts.pair_matrix --analyze --out data/benchmarks/gw_pair_matrix.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

from reblock.compare import load_permeability_config
from reblock.contracts import Block, Method
from reblock.data.pools import (
    DonorSkip,
    capetown_pool,
    evenly_spaced,
    fetch_donor_lines,
    iso_of,
    load_pools,
    overpass_footpaths,
    pbf_footpaths,
    zone_pool,
)
from reblock.data.settlements import exclusion_holdout
from reblock.derivations import access_before
from reblock.emit import pct_displaced
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.desire_lines import DesireLineSource
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    EgressContext,
    PermeabilityParams,
    permeability,
    with_width,
)
from reblock.transplant.gw import Arr
from reblock.transplant.operating_points import SIGNATURE, TRANSPORT
from reblock.transplant.signature import SignatureParams, signature, signature_distance
from reblock.transplant.snap import GapSnap, NearestNodeSnap
from reblock.transplant.transport import (
    TransportParams,
    fit_transport,
    parcel_xy,
    transport_lines,
)

CONF = Path("conf")


def _select_donor_candidates(
    recipient: Block,
    eligible: list[int],
    blocks: list[Block],
    signatures: Mapping[str, Arr],
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
        eligible, key=lambda j: signature_distance(signatures[blocks[j].block_id], r_sig))
    if n_candidates >= len(ranked):
        return ranked
    return [ranked[int(round(k))] for k in np.linspace(0, len(ranked) - 1, n_candidates)]


def rank1_distance_scaling(
    signatures: Mapping[str, Arr], sizes: list[int], *, n_trials: int = 200, seed: int = 0
) -> pd.DataFrame:
    """How the nearest-donor (rank-1) proxy-signature distance shrinks as the candidate donor pool
    grows, measured (not assumed from a theoretical N^(-1/d)) by resampling: for each pool size
    `N` in `sizes`, draw `n_trials` random (held-out recipient, N-block donor pool) splits from
    the pool signatures and record the nearest neighbour's distance. This is entirely a
    function of the cheap signature proxy (already computed for donor-candidate
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


@dataclass(frozen=True)
class Regression:
    """A within-recipient fixed-effects slope and its parametric significance."""

    beta: float
    se: float
    t: float
    dof: float
    p: float


@dataclass(frozen=True)
class DonorBootstrap:
    beta_median: float
    beta_lo95: float
    beta_hi95: float
    beta_sd: float
    beta_negative_frac: float
    n_boot: int


@dataclass(frozen=True)
class RangeRestriction:
    real_gw_dist_min: float
    real_gw_dist_max: float
    real_gw_dist_mean: float
    real_gw_dist_sd: float
    real_gw_dist_max_over_min: float
    feature_dist_min: float
    feature_dist_max: float
    feature_dist_max_over_min: float
    gw_feature_corr: float


@dataclass(frozen=True)
class FidelityAnalysis:
    n: int
    n_recipients: int
    pooled_pearson_r: float
    icc_perm_gap_by_recipient: float
    variance_explained_by_recipient: float
    within_recipient: Regression
    within_recipient_permutation_p: float
    donor_bootstrap: DonorBootstrap
    recipient_level_r: float
    within_recipient_excl_zero_length: Regression
    jackknife_beta_min: float
    jackknife_beta_max: float
    range_restriction: RangeRestriction


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


def within_recipient_regression(df: pd.DataFrame) -> Regression:
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
    return Regression(beta=beta, se=se, t=t_stat, dof=float(dof), p=p_value)


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


def range_restriction_summary(df: pd.DataFrame) -> RangeRestriction:
    """How restricted this sample's `real_gw_dist` range is relative to the `feature_dist` proxy
    used to STRATIFY donor selection -- range restriction attenuates a detectable correlation's
    MAGNITUDE, not whether an effect exists. `feature_dist` was deliberately stratified near/mid/
    far per recipient; if it tracked `real_gw_dist` perfectly, stratifying on one would stratify
    the other by the same ratio. A max/min ratio much narrower than feature_dist's own ratio, on
    the identical 100 pairs, shows the stratification did not transfer proportionally -- i.e. the
    achieved `real_gw_dist` range likely undersamples the full range achievable in the pool."""
    gw, fd = df["real_gw_dist"], df["feature_dist"]
    return RangeRestriction(
        real_gw_dist_min=float(gw.min()),
        real_gw_dist_max=float(gw.max()),
        real_gw_dist_mean=float(gw.mean()),
        real_gw_dist_sd=float(gw.std()),
        real_gw_dist_max_over_min=float(gw.max() / gw.min()),
        feature_dist_min=float(fd.min()),
        feature_dist_max=float(fd.max()),
        feature_dist_max_over_min=float(fd.max() / fd.min()),
        gw_feature_corr=float(gw.corr(fd)),
    )


def donor_bootstrap(
    df: pd.DataFrame, *, n_boot: int = 4000, seed: int = 0
) -> DonorBootstrap:
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
        betas.append(within_recipient_regression(sub).beta)
    b = np.asarray(betas, dtype=np.float64)
    return DonorBootstrap(
        beta_median=float(np.median(b)),
        beta_lo95=float(np.percentile(b, 2.5)),
        beta_hi95=float(np.percentile(b, 97.5)),
        beta_sd=float(b.std()),
        beta_negative_frac=float((b < 0).mean()),
        n_boot=len(b),
    )


def analyze_fidelity_vs_distance(
    df: pd.DataFrame, *, n_perm: int = 5000, seed: int = 0
) -> FidelityAnalysis:
    """The full clustering-aware analysis of an already-scored pair-matrix parquet: the naive
    pooled correlation, the recipient-clustering ICC that explains why pooling it is unsafe, the
    within-recipient fixed-effects slope with its parametric and permutation significance, the
    recipient-level aggregate correlation (the pooled number's cancelling counterpart), a
    robustness check dropping the zero-road-length rows, a leave-one-recipient-out jackknife of
    the within-recipient slope, and the range-restriction summary. Never re-fits any GW pair --
    purely a function of an already-scored matrix's columns."""
    _observed, perm_p = within_recipient_permutation_test(df, n_perm=n_perm, seed=seed)
    recipient_r, n_recipients = recipient_level_correlation(df)
    jackknife = [
        within_recipient_regression(df[df["recipient"] != rid]).beta
        for rid in df["recipient"].unique()
    ]
    perm_gap = df["perm_gap"].to_numpy(dtype=np.float64)
    return FidelityAnalysis(
        n=len(df),
        n_recipients=n_recipients,
        pooled_pearson_r=float(df["real_gw_dist"].corr(df["perm_gap"])),
        icc_perm_gap_by_recipient=icc_one_way(perm_gap, df["recipient"].to_numpy()),
        variance_explained_by_recipient=variance_explained_by_recipient(
            perm_gap, df["recipient"].to_numpy()),
        within_recipient=within_recipient_regression(df),
        within_recipient_permutation_p=perm_p,
        donor_bootstrap=donor_bootstrap(df, seed=seed),
        recipient_level_r=recipient_r,
        # pandas-stubs resolves a boolean-mask row filter to the Series overload.
        within_recipient_excl_zero_length=within_recipient_regression(
            cast(pd.DataFrame, df[df["road_len_m"] > 0])),
        jackknife_beta_min=float(min(jackknife)),
        jackknife_beta_max=float(max(jackknife)),
        range_restriction=range_restriction_summary(df),
    )


def _print_analysis(result: FidelityAnalysis) -> None:
    within, within_nz = result.within_recipient, result.within_recipient_excl_zero_length
    restriction, bs = result.range_restriction, result.donor_bootstrap
    print(f"n = {result.n} rows, {result.n_recipients} recipients")
    print(
        f"pooled Pearson r(real_gw_dist, perm_gap) = {result.pooled_pearson_r:.4f}  "
        "<-- artifact, see below"
    )
    print(f"ICC(1) (perm_gap by recipient, unbalanced-corrected) = "
          f"{result.icc_perm_gap_by_recipient:.4f}")
    print(f"  R^2 / eta^2 (raw SSB/SST, uncorrected)             = "
          f"{result.variance_explained_by_recipient:.4f}")
    print(
        f"within-recipient beta (perm_gap ~ real_gw_dist)  = {within.beta:.4f}  "
        f"SE={within.se:.4f}  t={within.t:.4f}  dof={within.dof:.0f}"
    )
    print(f"  p (t-distribution)      = {within.p:.4f}")
    print(f"  p (cluster permutation) = {result.within_recipient_permutation_p:.4f}")
    print(
        f"  DONOR BOOTSTRAP ({bs.n_boot:.0f} resamples) -- read this, not the point estimate:\n"
        f"    beta 95% interval [{bs.beta_lo95:+.3f}, {bs.beta_hi95:+.3f}]  "
        f"median {bs.beta_median:+.3f}  sd {bs.beta_sd:.3f}\n"
        f"    negative in {bs.beta_negative_frac:.1%} of donor resamples"
    )
    print(
        f"recipient-level aggregate r (n={result.n_recipients}) = "
        f"{result.recipient_level_r:.4f}  <-- the cancelling counterpart"
    )
    print(
        f"within-recipient beta, excl. zero-length rows = {within_nz.beta:.4f}  "
        f"p={within_nz.p:.4f}"
    )
    print(
        "jackknife beta range (leave-one-recipient-out) = "
        f"[{result.jackknife_beta_min:.4f}, {result.jackknife_beta_max:.4f}]"
    )
    print("range restriction:")
    print(
        f"  real_gw_dist: min={restriction.real_gw_dist_min:.4f} "
        f"max={restriction.real_gw_dist_max:.4f} mean={restriction.real_gw_dist_mean:.4f} "
        f"sd={restriction.real_gw_dist_sd:.4f}"
    )
    print(f"    max/min = {restriction.real_gw_dist_max_over_min:.2f}x")
    print(
        f"  feature_dist: min={restriction.feature_dist_min:.4f} "
        f"max={restriction.feature_dist_max:.4f}"
    )
    print(f"    max/min = {restriction.feature_dist_max_over_min:.2f}x")
    print(f"  corr(real_gw_dist, feature_dist) = {restriction.gw_feature_corr:.4f}")


# --- Scoring ------------------------------------------------------------------------------------


class PairRow(TypedDict):
    """One matrix row, exactly as written to the parquet."""

    recipient: str
    donor: str
    donor_type: str
    recipient_depth: float
    donor_depth: float
    real_gw_dist: float
    feature_dist: float
    perm_gap: float
    perm_proposal: float
    perm_direct: float
    displacement_proposal: float
    displacement_direct: float
    road_len_m: float
    wall_clock_s: float


@dataclass
class StageTimings:
    """Seconds spent per pipeline stage, accumulated over a run."""

    osm_fetch: float
    gw: float
    transplant: float
    clearance: float
    permeability: float


@dataclass(frozen=True)
class PairScorer:
    """Everything a matrix row is computed with, resolved once in `main`."""

    transport: TransportParams
    signature: SignatureParams
    snap: GapSnap
    direct: Method
    permeability: PermeabilityParams
    road_width_m: float     # stamped on the transplant; the direct method stamps its own

    def score(self, recipient: Block, donor: Block, donor_lines: gpd.GeoDataFrame,
              timings: StageTimings) -> PairRow:
        """One matrix row. `donor_lines` is the donor's material, in the donor's CRS."""
        t0 = time.time()
        r_xy, d_xy = parcel_xy(recipient), parcel_xy(donor)

        t = time.time()
        fit = fit_transport(d_xy, r_xy, self.transport)
        timings.gw += time.time() - t

        t = time.time()
        warped = transport_lines(donor_lines, fit, crs=recipient.crs)
        # Transplanted linework is bare geometry, so stamp the road width the metric requires.
        moved = with_width(self.snap.snap(warped, recipient), self.road_width_m)
        timings.transplant += time.time() - t

        t = time.time()
        road_len = float(moved.geometry.length.sum())
        direct = self.direct.propose(recipient).roads
        assert direct is not None, "the direct baseline always proposes a road frame"
        # Length-match the baseline by truncating to a prefix of comparable total length.
        cum = direct.geometry.length.cumsum()
        direct = cast(gpd.GeoDataFrame,
                      direct[cum <= road_len] if road_len > 0 else direct.iloc[:0])
        timings.clearance += time.time() - t

        t = time.time()
        ctx = EgressContext.of(recipient, self.permeability)
        perm_prop = permeability(ctx, moved)
        perm_direct = permeability(ctx, direct)
        timings.permeability += time.time() - t

        return PairRow(
            recipient=recipient.block_id,
            donor=donor.block_id,
            donor_type="osm_footpaths",
            # Depth is REPORTED, never gated on. It is the direct measure of the access problem
            # reblocking exists to fix, and the screen (`density_compactness` = n/P^2) does not
            # capture it -- measured on Cape Town, only 29% of the screened pool reaches k>=4,
            # where the old hand-rolled band required it of everything. Carrying it as a column is
            # what lets the analysis ask whether transplant fidelity depends on depth; a gate
            # would have made that question unanswerable from the matrix.
            recipient_depth=float(access_before(recipient).max()),
            donor_depth=float(access_before(donor).max()),
            real_gw_dist=fit.gw_dist,
            feature_dist=signature_distance(signature(d_xy, self.signature),
                                            signature(r_xy, self.signature)),
            perm_gap=float(perm_prop - perm_direct),
            perm_proposal=float(perm_prop),
            perm_direct=float(perm_direct),
            displacement_proposal=pct_displaced(moved, recipient.buildings),
            displacement_direct=pct_displaced(direct, recipient.buildings),
            road_len_m=road_len,
            wall_clock_s=time.time() - t0,
        )


def pair_scorer() -> PairScorer:
    """The published pair matrix's scorer: the transplant at its operating point, nearest-node
    snapping, and the shipped clearance preset as the direct baseline."""
    return PairScorer(
        transport=TRANSPORT, signature=SIGNATURE, snap=NearestNodeSnap(ChordSubstrate()),
        direct=ClearanceReblocker(substrate=ChordSubstrate(), repulsion=0.0, depth_target=2,
                                  max_roads=400, road_width_m=DEFAULT_ROAD_WIDTH_M),
        permeability=load_permeability_config(CONF).params, road_width_m=DEFAULT_ROAD_WIDTH_M)


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

    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    print("loading pools...")
    t_load = time.time()
    pools = load_pools(zone_pool(CONF, args.utm_zone) if args.utm_zone else capetown_pool(CONF))
    blocks, blocks_gdf = pools.blocks, pools.blocks_gdf
    # The cheap proxy, once for the whole pool: it only stratifies donor candidates by
    # similarity before a real GW fit is paid for, and is never written in place of it.
    signatures = {b.block_id: signature(parcel_xy(b), SIGNATURE) for b in blocks}
    where = f"UTM {args.utm_zone}" if args.utm_zone else "Cape Town"
    print(f"  {len(blocks)} screened {where} blocks in {time.time() - t_load:.1f}s")

    if args.rank1_scaling is not None:
        sizes = [int(s) for s in args.rank1_scaling.split(",")]
        df = rank1_distance_scaling(signatures, sizes)
        medians = df.groupby("pool_size")["rank1_dist"].median()
        print(medians)
        log_n, log_d = np.log(medians.index.to_numpy()), np.log(medians.to_numpy())
        slope, _intercept = np.polyfit(log_n, log_d, 1)
        print(f"fitted exponent (slope of log(rank1_dist) ~ log(pool_size)): {slope:.4f}")
        return

    n_recipients = max(1, -(-args.pairs // args.donors_per_recipient))  # ceil
    # Recipients come from the SCREENED set, donors from the donatable set -- different roles,
    # different requirements. See Pools.
    parcel_counts = [float(len(b.parcels)) for b in blocks]
    recipient_idx = evenly_spaced(pools.recipients, parcel_counts, n_recipients)
    donor_set = set(pools.donors)

    source: DesireLineSource = (pbf_footpaths(iso_of(blocks)) if args.desire_source == "pbf"
                                else overpass_footpaths())
    scorer = pair_scorer()
    donor_cache: dict[str, gpd.GeoDataFrame | DonorSkip] = {}

    # Resume support: this process has no reliable long-lived background execution in this
    # environment (a prior run_in_background attempt was killed with no trace and no
    # notification), so a run long enough to need the full ~90 s Bash-tool timeout budget must be
    # split into several bounded foreground invocations. Each successful row is checkpointed to
    # `args.out` immediately (below) so a kill loses at most the row in flight, and re-running the
    # same command picks up where it left off rather than re-scoring (and re-hitting Overpass for)
    # pairs already on disk. Never applies to --timing-only, which is a throwaway measurement.
    rows: list[PairRow] = []
    done_pairs: set[tuple[str, str]] = set()
    if not args.timing_only and args.out.exists():
        # This script's own output, so its records are the `PairRow`s it wrote.
        rows = cast(list[PairRow], pd.read_parquet(args.out).to_dict("records"))
        done_pairs = {(r["recipient"], r["donor"]) for r in rows}
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

    timings = StageTimings(osm_fetch=0.0, gw=0.0, transplant=0.0, clearance=0.0,
                           permeability=0.0)
    n_new = 0
    t0 = time.time()

    for i in recipient_idx:
        if len(rows) >= args.pairs:
            break
        recipient = blocks[i]
        eligible = [j for j in exclusion_holdout(blocks_gdf, i, radius_m=args.exclusion_radius_m)
                    if j in donor_set]
        if not eligible:
            continue
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
                fetched = fetch_donor_lines(source, donor)
                timings.osm_fetch += time.time() - fetch_t0
                donor_cache[donor.block_id] = fetched
                if isinstance(fetched, DonorSkip):
                    _bump_skip(fetched.value)
                    print(f"    skip donor {donor.block_id}: {fetched.value}")
            lines = donor_cache[donor.block_id]
            if isinstance(lines, DonorSkip):
                continue
            try:
                row = scorer.score(recipient, donor, lines, timings)
            except Exception as exc:  # noqa: BLE001 -- one bad pair must not sink a long run
                _bump_skip("scoring_error")
                print(f"    skip pair ({recipient.block_id}, {donor.block_id}): {exc!r}")
                continue
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
    for stage, secs in asdict(timings).items():
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
