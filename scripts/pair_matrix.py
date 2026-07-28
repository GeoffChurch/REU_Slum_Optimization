"""GW pair-matrix benchmark (Phase 1, unit 1d).

Per (recipient, donor): fit real entropic GW, transplant the donor's linework, snap it to the
recipient's substrate, and score it against a length-matched direct clearance solve. The output
parquet is a retrieval benchmark -- any future featurization or donor material can be scored
against it without re-solving anything.

`load_pools()` reads real Cape Town blocks directly from
``~/.cache/reblock/{blocks,buildings}_capetown_full.parquet`` -- the plan's census -> shortlist ->
provisioned-building-points chain never ran (Task 5 needs a 417 MB Geofabrik PBF not on this
machine; Task 7's provisioning was implemented but never executed), so there is no shortlist to
read instead. See docs/superpowers/notes/2026-07-27-gw-pair-matrix-findings.md for the full
writeup, and docs/superpowers/notes/2026-07-23-ot-road-transplant.md for the GW+UOT mechanism this
script drives.

Usage (module form -- puts the repo root on sys.path so `reblock.data.provision`'s
`from scripts.fetch_kblock_fixtures import ...` resolves; see
`scripts/fetch_desire_lines_snapshot.py` for the same convention):
    pixi run python -m scripts.pair_matrix --pairs 20 --timing-only
    pixi run python -m scripts.pair_matrix --pairs 100 --out data/benchmarks/gw_pair_matrix.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.error import URLError

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

_OT_DIR = Path("scratchpad/ot")
if not _OT_DIR.is_dir():
    raise SystemExit(
        "scratchpad/ot/ is missing. That directory holds the salvaged 2026-07-23 GW/transplant "
        "spike (ot_gw.py, transplant.py, select_donor.py) this script depends on -- it is "
        "gitignored scratchpad, never repo content, so it does not travel with a fresh checkout. "
        "Rebuild it from docs/superpowers/notes/2026-07-23-ot-road-transplant.md §1 (entropic "
        "GW: projected-gradient outer loop + log-domain Sinkhorn inner, eps=0.01, tau=1.0) before "
        "running this script."
    )
sys.path.insert(0, str(_OT_DIR))
from ot_gw import gw_cost  # noqa: E402
from select_donor import signature  # noqa: E402
from transplant import (  # noqa: E402
    _normalized_dist_matrix,
    fit_transport,
    gap_snap,
    transport_lines,
)

from reblock.budget import building_radii, displacement  # noqa: E402
from reblock.contracts import Block  # noqa: E402
from reblock.data.provision import cached_kblock_source  # noqa: E402
from reblock.data.settlements import exclusion_holdout  # noqa: E402
from reblock.methods.clearance import ClearanceReblocker  # noqa: E402
from reblock.methods.desire_lines import OSMDesireLines  # noqa: E402
from reblock.methods.osm_footpaths import interior_desire_lines  # noqa: E402
from reblock.permeability import permeability  # noqa: E402

CORRIDOR_M = 3.0
DEFAULT_CACHE = Path.home() / ".cache" / "reblock"
MIN_BUILDING_COUNT = 60
MAX_BUILDING_COUNT = 300
MIN_K_COMPLEXITY = 4
# select_donor.signature's fixed subsample size (N_SUB=50); a block with fewer real parcels than
# this can't be signed at all (ValueError), so it is dropped from the pool rather than the
# eligibility/selection logic having to special-case it downstream.
MIN_PARCELS = 50


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
    city: str = "capetown", *, min_buildings: int = 30
) -> tuple[list[Block], gpd.GeoDataFrame, dict[str, np.ndarray]]:
    """Real Cape Town blocks WITH building points, so `KblockSource` can build real `Block`s with
    Voronoi parcels (required by `gap_snap` and `permeability`) -- read directly from the cached
    full-city parquets rather than from a shortlist (see module docstring: the plan's shortlist
    chain never ran).

    The qualified pool is `building_count` in [60, 300] AND `k_complexity` >= 4, matching the
    RESCOPE note (1,136 Cape Town blocks against the parquet on this machine), further restricted
    to blocks whose real building-point join yields >= `MIN_PARCELS` parcels (a small fraction of
    the 1,136 -- the stored `building_count` column is a proxy for the real join, and
    `select_donor.signature`'s fixed subsample size needs a floor under it).

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
    raw = pd.read_parquet(
        DEFAULT_CACHE / f"blocks_{city}_full.parquet",
        columns=["block_id", "k_complexity", "building_count"],
    )
    qualified = raw[
        (raw.building_count >= MIN_BUILDING_COUNT)
        & (raw.building_count <= MAX_BUILDING_COUNT)
        & (raw.k_complexity >= MIN_K_COMPLEXITY)
    ]
    ids = sorted(qualified["block_id"].astype(str).tolist())

    src = cached_kblock_source(city, block_ids=ids, min_buildings=min_buildings)
    blocks = [b for b in src.region().blocks if len(b.parcels) >= MIN_PARCELS]
    blocks.sort(key=lambda b: b.block_id)
    if not blocks:
        raise SystemExit(f"load_pools: no qualified {city} blocks survived construction")

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
        signatures[b.block_id] = signature(xy)
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


def _donor_bbox_wgs84(donor: Block) -> tuple[float, float, float, float]:
    b = gpd.GeoSeries([donor.boundary], crs=donor.crs).to_crs(4326).total_bounds
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def fetch_donor_lines(
    source: OSMDesireLines, donor: Block, *, max_tries: int = 4, base_backoff_s: float = 2.0
) -> tuple[str, gpd.GeoDataFrame | None]:
    """Donor material: the donor block's real interior OSM footpaths (`donor_type =
    "osm_footpaths"`), mirroring `OsmFootpathsReblocker.propose`. Overpass is flaky right now
    (repeated 504 Gateway Timeouts observed during the coverage spike); a live fetch (cache miss)
    is retried with exponential backoff, and a donor whose OSM can't be fetched after `max_tries`
    is reported as `"fetch_failed"` -- the caller must skip it and count the skip, never silently
    drop it from the totals. `"empty_interior"` means the fetch succeeded but the donor has no
    interior footpath material once perimeter-retracing streets are subtracted -- also a skip, not
    a zero-length row."""
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
    r_xy = np.c_[recipient.parcels.geometry.centroid.x, recipient.parcels.geometry.centroid.y]
    d_xy = np.c_[donor.parcels.geometry.centroid.x, donor.parcels.geometry.centroid.y]

    t = time.time()
    result = fit_transport(d_xy, r_xy, eps=0.01, tau=1.0)
    timings["gw"] += time.time() - t
    dist = gw_cost(result.pi, _normalized_dist_matrix(d_xy), _normalized_dist_matrix(r_xy))

    t = time.time()
    warped = transport_lines(donor_lines, result, out_crs=recipient.crs)
    moved = gap_snap(warped, recipient)
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
        "real_gw_dist": float(dist),
        "feature_dist": float(np.linalg.norm(signature(d_xy) - signature(r_xy))),
        "perm_gap": float(perm_prop - perm_direct),
        "perm_proposal": float(perm_prop),
        "perm_direct": float(perm_direct),
        "displacement_proposal": displacement_fraction(recipient, moved),
        "displacement_direct": displacement_fraction(recipient, direct),
        "road_len_m": road_len,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--exclusion-radius-m", type=float, default=2000.0)
    ap.add_argument("--donors-per-recipient", type=int, default=5)
    ap.add_argument(
        "--candidate-multiplier",
        type=int,
        default=3,
        help="try up to this many x donors-per-recipient candidates per recipient, backfilling "
        "around OSM fetch/empty-interior skips",
    )
    ap.add_argument("--out", type=Path, default=Path("data/benchmarks/gw_pair_matrix.parquet"))
    ap.add_argument(
        "--rank1-scaling",
        type=str,
        default=None,
        help="comma-separated pool sizes (e.g. '10,30,100,300,1000'); if set, run ONLY the "
        "rank-1-distance pool-size scaling analysis (cheap, signature-proxy-only) and exit "
        "without touching GW/OSM/clearance at all",
    )
    args = ap.parse_args()

    print("loading pools...")
    t_load = time.time()
    blocks, blocks_gdf, signatures = load_pools()
    print(f"  {len(blocks)} qualified Cape Town blocks in {time.time() - t_load:.1f}s")

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

    source = OSMDesireLines()
    donor_cache: dict[str, tuple[str, gpd.GeoDataFrame | None]] = {}
    skip_counts: Counter[str] = Counter()

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
                    skip_counts[status] += 1
                    print(f"    skip donor {donor.block_id}: {status}")
            status, lines = donor_cache[donor.block_id]
            if status != "ok" or lines is None:
                continue
            pair_t0 = time.time()
            try:
                row = score_pair(recipient, donor, lines, timings)
            except Exception as exc:  # noqa: BLE001 -- one bad pair must not sink a long run
                skip_counts["scoring_error"] += 1
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
    print(f"skips this run: {dict(skip_counts)}")
    if args.timing_only:
        print("(timing-only: nothing written)")
    else:
        print(f"total rows on disk: {len(rows)}")


if __name__ == "__main__":
    main()
