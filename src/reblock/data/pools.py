"""Research pools: screened recipients and donatable donors, materialized as real Blocks.

The OT-transplant benchmarks (`scripts/pair_matrix.py`, `scripts/consensus_*.py`), the mimicry
scorers and the `scripts/perf/` studies all draw their blocks here, so each is measured on the same
population -- and that population is selected through the shipped `Source -> Screen ->
RegionBuilder` stages, built from conf/ by `reblock.presets`, not a private band. A private
`building_count in [60,300] AND k_complexity >= 4` band came first and made none of the OT numbers
comparable to any shipped method's (docs/superpowers/notes/2026-07-28-slope-is-pool-dependent.md).

The screen is `density_compactness` (n/P^2) at its calibrated ABSOLUTE floor, not a percentile: a
percentile re-defines the population whenever the corpus changes, and the pool is meant to scale
from Cape Town to the ZAF+KEN shortlist, where the same percentile is a four-times different cut.
It is peel-free, so selecting never builds a Block. Depth is deliberately NOT gated on: it is the
direct measure of the access problem, so it belongs in a benchmark's OUTPUT as a stratifier, and a
gate would make "does fidelity depend on depth?" unanswerable.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast
from urllib.error import URLError

import geopandas as gpd
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.data.counts import RAW_COUNT
from reblock.data.kblock import KblockSource
from reblock.data.osm_extract import FOOTPATH_TAGS, PbfDesireLines, utm_zone_epsg
from reblock.data.provision import DEFAULT_CACHE
from reblock.methods.desire_lines import DesireLineSource, OSMDesireLines
from reblock.methods.osm_footpaths import interior_desire_lines
from reblock.presets import Stages, load_stages

log = logging.getLogger(__name__)

# compare_config under these overrides is the pool's three stages: Cape Town's full-city parquets
# at spacing discs, the density_compactness screen at its absolute floor counting Open Buildings
# points, and singleton regions. Every group is spelled, defaults included, so a change to
# compare_config's defaults cannot move the pool unannounced.
_CAPETOWN = ("data=capetown_full", "data.min_buildings=30", "buildings=spacing",
             "screen=dense_compact", "screen.min_buildings=30", "metric=density_compactness",
             "building_count=open_buildings", "proxy_keep_n=1000", "region_builder=identity")

SHORTLIST_BLOCKS = DEFAULT_CACHE / "blocks_shortlist.parquet"
SHORTLIST_BUILDINGS = DEFAULT_CACHE / "buildings_shortlist.parquet"

PBF_BY_ISO = {"ZAF": "south-africa-latest.osm.pbf", "KEN": "kenya-latest.osm.pbf"}


@dataclass(frozen=True)
class PoolSpec:
    """What a pool is drawn from, and the bounds it is drawn within."""

    stages: Stages
    census_dir: Path            # holds osm_coverage_{ISO}.parquet, the donor-eligibility census
    # COMPUTE bounds on the source's stored building count, not a quality judgement -- what a GW
    # fit can afford, since it is quadratic in parcels. What is worth reblocking is the screen's
    # call. The stored count is only a proxy for the real building-point join, hence:
    min_building_count: int
    max_building_count: int
    min_parcels: int            # real parcels after the join; a donor signature subsamples 50
    min_interior_m: float       # interior footpath a donor must carry, per the census


def capetown_config(config_dir: Path) -> DictConfig:
    """`compare_config` composed into the Cape Town pool's stages. Building them provisions the
    city data; composing does not."""
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        return compose(config_name="compare_config", overrides=list(_CAPETOWN))


def capetown_pool(config_dir: Path) -> PoolSpec:
    """The pool every committed pair-matrix, consensus and perf result was drawn from."""
    return PoolSpec(stages=load_stages(capetown_config(config_dir)), census_dir=DEFAULT_CACHE,
                    min_building_count=60, max_building_count=300, min_parcels=50,
                    min_interior_m=100.0)


def zone_source(epsg: int, like: KblockSource) -> KblockSource:
    """A `KblockSource` over the provisioned ZAF+KEN shortlist, restricted to ONE UTM zone, with
    `like`'s building floor and tier.

    The restriction is not optional. `KblockSource.region()` calls `estimate_utm_crs()` on the
    WHOLE blocks frame -- deliberately, so the CRS stays stable under `block_ids` filtering -- so
    a two-country shortlist would hand every block one UTM zone and distort area, perimeter and
    every distance far from that meridian. Filtering by `block_ids` cannot fix that, precisely
    because of the stability guarantee; the zone subset has to be its own parquet, materialized
    here once and then cached.

    Splitting by zone costs the cross-zone donor pairs. Geographic distance was measured to be
    uninformative about GW distance, so that is a real loss, but each zone is a different metro,
    which makes zone-wise runs a replication rather than merely a smaller sample.
    """
    for path in (SHORTLIST_BLOCKS, SHORTLIST_BUILDINGS):
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path} -- run `python -m scripts.provision_shortlist` first")
    zone_path = SHORTLIST_BLOCKS.with_name(f"blocks_shortlist_z{epsg}.parquet")
    if not zone_path.exists():
        frame = gpd.read_parquet(SHORTLIST_BLOCKS)
        rep = frame.geometry.representative_point()
        keep = np.array([utm_zone_epsg(x, y) == epsg
                         for x, y in zip(rep.x, rep.y, strict=True)])
        subset = cast(gpd.GeoDataFrame, frame[keep].reset_index(drop=True))
        if subset.empty:
            raise ValueError(f"no shortlist blocks in UTM zone {epsg}")
        subset.to_parquet(zone_path)
        log.info("materialized %s: %d blocks", zone_path.name, len(subset))
    return KblockSource(zone_path, SHORTLIST_BUILDINGS, region_id=f"shortlist-z{epsg}",
                        min_buildings=like.min_buildings, block_ids=None,
                        building_tier=like.building_tier, member_buildings=None)


def zone_pool(config_dir: Path, epsg: int) -> PoolSpec:
    """`capetown_pool`'s screen and bounds over one UTM zone of the ZAF+KEN shortlist."""
    base = capetown_pool(config_dir)
    return replace(base, stages=replace(base.stages,
                                        source=zone_source(epsg, _kblock(base.stages))))


@dataclass(frozen=True)
class Pools:
    """The materialized pool, with the two ROLES kept apart.

    A recipient is a reblocking target: it needs building points and parcels, and should be
    whatever the screen says is worth reblocking. A donor is material to transplant: it needs
    interior footpaths, and nothing about being dense-and-compact is required of it.

    Screening both roles by n/P^2 starved every early run of the only variable that mattered.
    Donors selected that way are near-identical to their recipients, so GW distance barely varied
    (range ratios 8.96x / 4.54x / 1.71x across three 500-pair runs) and the slope tracked that
    range. Decoupling the roles widened the corpus-wide donor pool 57x (247 -> 14,189).
    """

    blocks: list[Block]             # sorted by block_id
    blocks_gdf: gpd.GeoDataFrame    # block boundaries in `blocks` order, for `exclusion_holdout`
    recipients: list[int]           # indices into `blocks`
    donors: list[int]


def _kblock(stages: Stages) -> KblockSource:
    if not isinstance(stages.source, KblockSource):
        raise TypeError(f"a pool reads kblock parquet columns; got {type(stages.source).__name__}")
    return stages.source


def donatable_ids(census_dir: Path, iso: str, min_interior_m: float) -> set[str]:
    """Blocks the census says carry at least `min_interior_m` of interior footpath.

    Read from the census rather than discovered by fetching: a fetch per candidate is what made
    donor slots scarce (509 `empty_interior` skips for 68 usable pairs in Gauteng), and the census
    already measured every block. A block absent from it failed the census's own prefilter
    (k_complexity >= 3, building_count >= 40) -- shallow or tiny, not donor material.
    """
    path = census_dir / f"osm_coverage_{iso}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run `python -m scripts.osm_census --iso {iso}` first; donor "
            f"eligibility is read from the census, not discovered by fetching")
    cen = pd.read_parquet(path, columns=["block_id", "census_failed", "interior_length_m_0.5"])
    ok = (~cen["census_failed"]) & (cen["interior_length_m_0.5"] >= min_interior_m)
    return {str(b) for b in cen["block_id"][ok]}


def load_pools(spec: PoolSpec) -> Pools:
    """Screen, bound, grow and build the pool.

    Recipients are what the screen flags inside the compute band; donors are what the census says
    is donatable inside it. The region builder grows the recipients' singleton seed groups -- a
    no-op for the identity builder, which is the point: the pool rides the same stages as every
    shipped method, so an accreting builder (which street-form donor material needs, a single
    block having no internal streets) is a substitution here, not a rewrite. A builder returning
    non-singleton groups would need `region.region_block` to fuse each group before scoring; that
    is not built until something needs it.
    """
    source = _kblock(spec.stages)
    frame = source.block_geometries()
    counts = dict(zip(frame["block_id"], frame[RAW_COUNT], strict=True))
    selected = spec.stages.screen.select(source)
    flagged = set(counts) if selected is None else set(selected)
    in_band = {b for b, c in counts.items()
               if spec.min_building_count <= float(c) <= spec.max_building_count}
    recipient_ids = sorted(flagged & in_band)
    if not recipient_ids:
        raise ValueError(f"{source.region_id}: the screen flagged nothing inside the "
                         f"[{spec.min_building_count}, {spec.max_building_count}] building band")
    iso = recipient_ids[0].split(".", 1)[0]
    donor_ids = sorted(donatable_ids(spec.census_dir, iso, spec.min_interior_m) & in_band)
    log.info("screen flagged %d -> %d recipients; %d donors with >= %.0f m interior footpath",
             len(flagged), len(recipient_ids), len(donor_ids), spec.min_interior_m)

    groups = spec.stages.region_builder.build(
        cast(gpd.GeoDataFrame, frame[frame["block_id"].isin(recipient_ids)]),
        [[b] for b in recipient_ids], depth_fn=None)
    ids = sorted({b for group in groups for b in group} | set(donor_ids))

    # The source narrowed to `ids`, keeping its own tier and member buildings.
    narrowed = KblockSource(source.blocks_path, source.buildings_path,
                            region_id=source.region_id, min_buildings=source.min_buildings,
                            block_ids=ids, building_tier=source.building_tier,
                            member_buildings=source.member_buildings)
    blocks = sorted((b for b in narrowed.region().blocks if len(b.parcels) >= spec.min_parcels),
                    key=lambda b: b.block_id)
    if not blocks:
        raise ValueError(f"{source.region_id}: no pool block survived construction")
    r_set, d_set = set(recipient_ids), set(donor_ids)
    pools = Pools(
        blocks=blocks,
        blocks_gdf=gpd.GeoDataFrame({"block_id": [b.block_id for b in blocks]},
                                    geometry=[b.boundary for b in blocks], crs=blocks[0].crs),
        recipients=[i for i, b in enumerate(blocks) if b.block_id in r_set],
        donors=[i for i, b in enumerate(blocks) if b.block_id in d_set])
    log.info("materialized %d blocks: %d usable recipients, %d usable donors",
             len(blocks), len(pools.recipients), len(pools.donors))
    return pools


def evenly_spaced(idx: Sequence[int], key: Sequence[float], n: int) -> list[int]:
    """`n` of `idx` spanning `key`'s range -- sorted, then evenly spaced ranks from min to max.
    Not a random sample: the extremes are in by construction."""
    order = sorted(idx, key=lambda i: key[i])
    if n >= len(order):
        return order
    return [order[int(round(k))] for k in np.linspace(0, len(order) - 1, n)]


def iso_of(blocks: Sequence[Block]) -> str:
    """The one country a pool belongs to, from its kblock ids (`ZAF.9.3.1_1_44882`).

    Load-bearing: a PBF covers exactly its own extract, so pointing a Kenyan pool at the South
    Africa file does not error -- every donor just comes back with no interior footpaths. The first
    Nairobi run reported `empty_interior: 90` and zero pairs, flatly contradicted by the census.
    Deriving the extract from the data removes the chance to pick the wrong one by hand.
    """
    isos = {str(b.block_id).split(".", 1)[0] for b in blocks}
    if len(isos) != 1:
        raise ValueError(f"pool spans multiple countries {sorted(isos)}; one PBF cannot cover it")
    iso = isos.pop()
    if iso not in PBF_BY_ISO:
        raise ValueError(
            f"no Geofabrik extract configured for {iso!r}; known: {sorted(PBF_BY_ISO)}")
    return iso


def pbf_path(iso: str) -> Path:
    """The local Geofabrik extract for `iso`."""
    path = DEFAULT_CACHE / "osm_pbf" / PBF_BY_ISO[iso]
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- download it from https://download.geofabrik.de/, or read "
            f"footpaths from the live Overpass API instead")
    return path


def pbf_footpaths(iso: str) -> PbfDesireLines:
    """Footpaths from the local extract: one ~40 s read into memory, then a bbox window per block
    and no network. The default, because a 100-pair run against Overpass got 27 usable pairs and
    214 failed fetches, spending 100% of wall clock on the network."""
    return PbfDesireLines(pbf_path=pbf_path(iso), tags=FOOTPATH_TAGS)


def overpass_footpaths() -> OSMDesireLines:
    """Footpaths from the live Overpass API: any bbox on earth, but a shared third-party service.
    Not a fallback for `pbf_footpaths` -- the two cover disjoint ranges."""
    return OSMDesireLines(tags=FOOTPATH_TAGS, endpoint="https://overpass-api.de/api/interpreter",
                          cache_dir=None, snapshot=None, timeout_s=60.0)


class DonorSkip(StrEnum):
    """Why a block yielded no donor material. Counted by the scripts, never silently dropped."""

    FETCH_FAILED = "fetch_failed"       # the source failed every retry
    EMPTY_INTERIOR = "empty_interior"   # fetched, but nothing is left once streets are subtracted


def _bbox_wgs84(block: Block) -> tuple[float, float, float, float]:
    b = gpd.GeoSeries([block.boundary], crs=block.crs).to_crs(4326).total_bounds
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def fetch_donor_lines(source: DesireLineSource, block: Block, *, max_tries: int = 4,
                      base_backoff_s: float = 2.0) -> gpd.GeoDataFrame | DonorSkip:
    """The block's real interior footpaths, as `OsmFootpathsReblocker.propose` reads them -- but
    widthless: this is material (an alignment), and whoever builds a road on it stamps the width.

    Retries with exponential backoff, which a PBF source never needs and a network source
    (Overpass) does.
    """
    for attempt in range(max_tries):
        try:
            lines = source.desire_lines(_bbox_wgs84(block), block.crs)
            break
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            if attempt == max_tries - 1:
                return DonorSkip.FETCH_FAILED
            wait = base_backoff_s * (2 ** attempt)
            log.warning("retry %s footpath fetch in %.0fs (%r)", block.block_id, wait, exc)
            time.sleep(wait)
    else:
        raise ValueError(f"max_tries must be >= 1, got {max_tries}")
    interior = interior_desire_lines(lines, block.boundary,
                                     unary_union(list(block.streets.geometry)), block.crs)
    return DonorSkip.EMPTY_INTERIOR if interior.empty else interior
