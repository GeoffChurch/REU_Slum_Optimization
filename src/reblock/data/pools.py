"""Research pools: screened recipients and donatable donors, materialized as real Blocks.

The OT-transplant benchmarks (`scripts/pair_matrix.py`, `scripts/consensus_matrix.py`), the mimicry
scorers and the `scripts/perf/` studies all draw their blocks here, so each is measured on the same
population -- and that population is selected through the shipped `Source -> Screen ->
RegionBuilder` stages, built from conf/ by `reblock.presets`, not a private band. A private
`building_count in [60,300] AND k_complexity >= 4` band came first and made none of the OT numbers
comparable to any shipped method's (docs/superpowers/notes/2026-07-28-slope-is-pool-dependent.md).

A pool is CONFIGURED (`conf/donor_pool/`): Hydra builds its stages like any other strategy's, and
the pool only ever consumes them -- it composes no config itself, so a donor-driven method is as
ordinary inside a `@hydra.main` app as anywhere else. Scripts that are not Hydra apps compose the
preset at their own top level (`scripts/_donor_pool.py`).

The screen is `density_compactness` (n/P^2) at its calibrated ABSOLUTE floor, not a percentile: a
percentile re-defines the population whenever the corpus changes, and the pool is meant to scale
from Cape Town to the ZAF+KEN shortlist, where the same percentile is a four-times different cut.
It is peel-free, so selecting never builds a Block. Depth is deliberately NOT gated on: it is the
direct measure of the access problem, so it belongs in a benchmark's OUTPUT as a stratifier, and a
gate would make "does fidelity depend on depth?" unanswerable.
"""
from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import cast
from urllib.error import URLError

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from reblock.buildings import Extents, tier_identity
from reblock.contracts import Block
from reblock.data.counts import RAW_COUNT
from reblock.data.kblock import KblockSource
from reblock.data.osm_extract import FOOTPATH_TAGS, PbfDesireLines, utm_zone_epsg
from reblock.data.provision import DEFAULT_CACHE
from reblock.derive_graph import closure_hash, config_identity
from reblock.methods.desire_lines import DesireField, OSMDesireLines, mapped_field
from reblock.methods.osm_footpaths import (
    FootpathSource,
    block_bbox_wgs84,
    block_footpaths,
    interior_footpaths,
)
from reblock.presets import Stages

log = logging.getLogger(__name__)

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


def zone_source(epsg: int, *, cache_dir: Path, min_buildings: int,
                building_tier: Callable[[gpd.GeoDataFrame], Extents]) -> KblockSource:
    """A `KblockSource` over the provisioned ZAF+KEN shortlist in `cache_dir`, restricted to ONE
    UTM zone (`conf/donor_pool/shortlist_zone.yaml`).

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
    shortlist_blocks = cache_dir / "blocks_shortlist.parquet"
    shortlist_buildings = cache_dir / "buildings_shortlist.parquet"
    for path in (shortlist_blocks, shortlist_buildings):
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path} -- run `python -m scripts.provision_shortlist` first")
    zone_path = cache_dir / f"blocks_shortlist_z{epsg}.parquet"
    if not zone_path.exists():
        frame = gpd.read_parquet(shortlist_blocks)
        rep = frame.geometry.representative_point()
        keep = np.array([utm_zone_epsg(x, y) == epsg
                         for x, y in zip(rep.x, rep.y, strict=True)])
        subset = cast(gpd.GeoDataFrame, frame[keep].reset_index(drop=True))
        if subset.empty:
            raise ValueError(f"no shortlist blocks in UTM zone {epsg}")
        subset.to_parquet(zone_path)
        log.info("materialized %s: %d blocks", zone_path.name, len(subset))
    return KblockSource(zone_path, shortlist_buildings, region_id=f"shortlist-z{epsg}",
                        min_buildings=min_buildings, block_ids=None,
                        building_tier=building_tier, member_buildings=None)


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
    blocks_gdf: gpd.GeoDataFrame    # block boundaries in `blocks` order, for exclusion radii
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


@dataclass(frozen=True)
class PoolIds:
    """Which blocks a pool holds, by id: everything its screen, census and bounds decide, and
    nothing that needs a Block."""

    recipients: tuple[str, ...]     # screened, in the building band
    donors: tuple[str, ...]         # donatable per the census, in the band
    members: tuple[str, ...]        # every block to build: the grown recipient groups and donors


# One selection per distinct pool per process, as `_MATERIALIZED` holds one build: every arm of a
# study instantiates its own stages, and each arm's pool re-read the source's geometries and the
# census (six times over in the consensus study). Keyed on what the selection reads. The screen
# enters by what it flagged: it has no identity of its own, and asking it again is one lookup of
# its own derivation (`derivations.screen_selection`).
_SELECTED: dict[Hashable, PoolIds] = {}


def select_pool(spec: PoolSpec) -> PoolIds:
    """Screen, bound and grow the pool -- everything but building it.

    Recipients are what the screen flags inside the compute band; donors are what the census says
    is donatable inside it. The region builder grows the recipients' singleton seed groups -- a
    no-op for the identity builder, which is the point: the pool rides the same stages as every
    shipped method, so an accreting builder (which street-form donor material needs, a single
    block having no internal streets) is a substitution here, not a rewrite. A builder returning
    non-singleton groups would need `region.region_block` to fuse each group before scoring; that
    is not built until something needs it.
    """
    source = _kblock(spec.stages)
    selected = spec.stages.screen.select(source)
    builder = config_identity(spec.stages.region_builder)
    if builder is None:     # a builder holding something uncacheable: select afresh, as `derive`
        return _select(spec, source, selected)
    key = (_source_key(source), None if selected is None else tuple(selected), builder,
           spec.census_dir, spec.min_building_count, spec.max_building_count,
           spec.min_interior_m)
    if key not in _SELECTED:
        _SELECTED[key] = _select(spec, source, selected)
    return _SELECTED[key]


def _select(spec: PoolSpec, source: KblockSource, selected: list[str] | None) -> PoolIds:
    frame = source.block_geometries()
    counts = dict(zip(frame["block_id"], frame[RAW_COUNT], strict=True))
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
    members = sorted({b for group in groups for b in group} | set(donor_ids))
    return PoolIds(recipients=tuple(recipient_ids), donors=tuple(donor_ids),
                   members=tuple(members))


def build_pool(source: KblockSource, ids: PoolIds, min_parcels: int) -> Pools:
    """Build every member block (a Voronoi each) and keep those with at least `min_parcels`."""
    blocks = sorted((b for b in source.restricted(ids.members).region().blocks
                     if len(b.parcels) >= min_parcels),
                    key=lambda b: b.block_id)
    if not blocks:
        raise ValueError(f"{source.region_id}: no pool block survived construction")
    r_set, d_set = set(ids.recipients), set(ids.donors)
    pools = Pools(
        blocks=blocks,
        blocks_gdf=gpd.GeoDataFrame({"block_id": [b.block_id for b in blocks]},
                                    geometry=[b.boundary for b in blocks], crs=blocks[0].crs),
        recipients=[i for i, b in enumerate(blocks) if b.block_id in r_set],
        donors=[i for i, b in enumerate(blocks) if b.block_id in d_set])
    log.info("materialized %d blocks: %d usable recipients, %d usable donors",
             len(blocks), len(pools.recipients), len(pools.donors))
    return pools


def load_pools(spec: PoolSpec) -> Pools:
    """Select and build the pool `spec` describes."""
    return build_pool(_kblock(spec.stages), select_pool(spec), spec.min_parcels)


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


class DonorPool(ABC):
    """A materialized pool as configuration: what a donor-driven method draws its donors from, and
    what a study draws its recipients from.

    Materialized on first use, never at construction: materializing is a census read, a screen and
    a Voronoi per block, and a preset is built with everything else an entry point is configured
    with, whether or not the run ever proposes with it.
    """

    @abstractmethod
    def pools(self) -> Pools: ...

    @abstractmethod
    def footpaths(self) -> FootpathSource:
        """Where every block's own footpaths are read from: the donors' material, and the
        ground truth a recipient's prediction is scored against."""

    @property
    @abstractmethod
    def identity(self) -> Hashable | None:
        """The pool's CONTENT: every donor's block identity, in order, and the footpath source's.
        Which recipients the screen flags is not part of it -- no donor-driven method reads
        them."""

    @abstractmethod
    def load(self) -> None:
        """Materialize now rather than on first use, footpaths included: what a process pool forks
        after, so every worker shares one copy instead of each reading its own."""


@dataclass(frozen=True, eq=False)
class _Materialized:
    pools: Pools
    footpaths: PbfDesireLines
    identity: Hashable | None


# One materialization per distinct pool per process. Every method is built by its own `instantiate`
# call, so four study arms configured with one preset hold four equal-but-distinct stage objects;
# keyed on what the build reads, they share one pool (measured: a second Cape Town pool is +0.7 GB
# and 12 s, and one PBF read each). Like `derive_graph`'s L1, never persisted.
_MATERIALIZED: dict[Hashable, _Materialized] = {}


def _source_key(source: KblockSource) -> Hashable:
    """Everything `build_pool` reads off the source, by value."""
    return (source.blocks_path, source.buildings_path, source.region_id, source.min_buildings,
            source.block_ids,
            tier_identity(source.building_tier), source.member_buildings)


def _materialize(spec: PoolSpec) -> _Materialized:
    source = _kblock(spec.stages)
    ids = select_pool(spec)     # one selection per distinct pool (`_SELECTED`)
    key = (_source_key(source), ids, spec.min_parcels)
    if key not in _MATERIALIZED:
        pools = build_pool(source, ids, spec.min_parcels)
        footpaths = pbf_footpaths(iso_of(pools.blocks))
        donors = [pools.blocks[j].identity for j in pools.donors]
        identity = (None if any(d is None for d in donors) else
                    ("donor_pool", hashlib.sha256(repr(donors).encode()).hexdigest(),
                     footpaths.identity))
        _MATERIALIZED[key] = _Materialized(pools=pools, footpaths=footpaths, identity=identity)
    return _MATERIALIZED[key]


@dataclass(frozen=True)
class ScreenedPool(DonorPool):
    """The pool `load_pools` draws through a `PoolSpec`'s built stages, with footpaths from its own
    country's PBF (`iso_of`). The spec arrives built -- `conf/donor_pool/` -- and this only ever
    consumes it."""

    spec: PoolSpec

    @cached_property
    def _materialized(self) -> _Materialized:
        return _materialize(self.spec)

    def pools(self) -> Pools:
        return self._materialized.pools

    def footpaths(self) -> PbfDesireLines:
        return self._materialized.footpaths

    @property
    def identity(self) -> Hashable | None:
        return self._materialized.identity

    def load(self) -> None:
        # The extract is read on first use otherwise, and it is the one read worth sharing: a
        # country's PBF is ~10 s and ~1 GB of GDAL scratch per process that reads it.
        self._materialized.footpaths.lines()


@dataclass(frozen=True)
class PoolFootpaths:
    """A pool's own footpath source as a configurable desire source (`conf/desire_source/pool`):
    the ground truth a study scores predictions against then comes from the very source the
    donors' material does, and is read once."""

    pool: DonorPool

    @property
    def identity(self) -> Hashable | None:
        source = self.pool.footpaths().identity
        return None if source is None else (closure_hash(__name__), source)

    def footpaths(self, bbox_wgs84: tuple[float, float, float, float],
                  crs: CRS) -> gpd.GeoDataFrame:
        return self.pool.footpaths().footpaths(bbox_wgs84, crs)

    def desire_field(self, block: Block) -> DesireField:
        return mapped_field(block_footpaths(self, block))


class DonorSkip(StrEnum):
    """Why a block yielded no donor material. Counted by the scripts, never silently dropped."""

    FETCH_FAILED = "fetch_failed"       # the source failed every retry
    EMPTY_INTERIOR = "empty_interior"   # fetched, but nothing is left once streets are subtracted


def fetch_donor_lines(source: FootpathSource, block: Block, *, max_tries: int = 4,
                      base_backoff_s: float = 2.0) -> gpd.GeoDataFrame | DonorSkip:
    """The block's real interior footpaths, as `osm_footpaths.block_footpaths` reads them -- but
    widthless: this is material (an alignment), and whoever builds a road on it stamps the width.

    Retries with exponential backoff, which a PBF source never needs and a network source
    (Overpass) does.
    """
    for attempt in range(max_tries):
        try:
            lines = source.footpaths(block_bbox_wgs84(block), block.crs)
            break
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            if attempt == max_tries - 1:
                return DonorSkip.FETCH_FAILED
            wait = base_backoff_s * (2 ** attempt)
            log.warning("retry %s footpath fetch in %.0fs (%r)", block.block_id, wait, exc)
            time.sleep(wait)
    else:
        raise ValueError(f"max_tries must be >= 1, got {max_tries}")
    interior = interior_footpaths(lines, block)
    return DonorSkip.EMPTY_INTERIOR if interior.empty else interior
