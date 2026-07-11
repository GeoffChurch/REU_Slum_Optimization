"""Fixture builders shared by the scoring-equivalence harness (tests/test_scoring_equivalence.py)
and reused by later perf-refactor tasks. Reloads the 1808 sample block + its road sets (no
`propose()` calls) and pairs them with the reference (E, directness, curves, AUC) values pinned
in `tests/data/scoring/ref_values_1808.json`.

Also builds the three fixture families the design's "Correctness strategy" requires beyond 1808
(a coincident-entry case, a sparse straight chord, and the deep 2-block region), pinned in
`tests/data/scoring/ref_values_extra.json`. Values in both JSON files were captured from the
CURRENT (pre-refactor, verified-correct) `network_efficiency`/`efficiency_directness_curves`/`auc`
-- that output IS the ground truth this harness protects."""
import json
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
from pyproj import CRS
from shapely import wkt
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.data.kblock import KblockSource
from reblock.region import region_block

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REF: dict[str, dict[str, Any]] = json.loads(
    (_REPO_ROOT / "tests/data/scoring/ref_values_1808.json").read_text())
_REF_EXTRA: dict[str, dict[str, Any]] = json.loads(
    (_REPO_ROOT / "tests/data/scoring/ref_values_extra.json").read_text())

UTM = CRS.from_epsg(32643)


def _block_1808() -> Block:
    src = KblockSource(_REPO_ROOT / "tests/data/kblock/blocks_dji_sample.parquet",
                       _REPO_ROOT / "tests/data/kblock/buildings_dji_sample.parquet", "dji",
                       block_ids=["DJI.3_1_1808"])
    return next(iter(src.region().blocks))


def _roads(block: Block, ref: dict[str, dict[str, Any]], key: str) -> gpd.GeoDataFrame | None:
    r = ref[key]
    if "wkt" not in r:
        return None
    return gpd.GeoDataFrame(geometry=[wkt.loads(w) for w in r["wkt"]], crs=block.parcels.crs)


def _block_coincident() -> Block:
    """3 parcels, 1 short street segment `(0,0)-(1,0)`. Parcels 0 and 1 sit past the segment's
    RIGHT end (x > 1): `LineString.project` clamps a point beyond an endpoint to that endpoint,
    so both parcels' nearest-point-on-edge projection lands on the exact same node `(1.0, 0.0)`
    regardless of their (different) y-offsets -- a genuine coincident entry (`netdist == 0`
    between them), not a tolerance-boundary coincidence. Parcel 2 sits near the segment's
    interior and gets a distinct entry `(0.5, 0.0)`, so the fixture isn't degenerately all-one-
    node. Verified (see task report): `_line_entries` produces entries
    `[(1.0, 0.0), (1.0, 0.0), (0.5, 0.0)]` -- parcels 0 and 1 share a node."""
    street = LineString([(0.0, 0.0), (1.0, 0.0)])
    poly_a = Polygon([(1.1, 0.0), (1.3, 0.0), (1.3, 0.25), (1.1, 0.25)])
    poly_b = Polygon([(1.1, 0.3), (1.3, 0.3), (1.3, 0.55), (1.1, 0.55)])
    poly_c = Polygon([(0.3, 0.1), (0.7, 0.1), (0.7, 0.4), (0.3, 0.4)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1, 2]}, geometry=[poly_a, poly_b, poly_c],
                               crs=UTM)
    boundary = cast(Polygon, unary_union([poly_a, poly_b, poly_c]))
    streets = gpd.GeoDataFrame(geometry=[street], crs=UTM)
    return Block(block_id="coincident", crs=UTM, boundary=boundary, parcels=parcels,
                streets=streets)


def _block_sparse_chord() -> Block:
    """The line-proximity sparse-chord fixture (mirrors
    `test_budget.py::test_line_proximity_scores_a_sparse_straight_chord`): a deep 3x7 grid block
    with bottom-only street frontage, reblocked with a single bare 2-point straight chord --
    only its endpoints are graph vertices, so this exercises nearest-POINT (not nearest-vertex)
    entry projection."""
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(7)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    streets = gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (3.0, 0.0)])], crs=UTM)
    return Block(block_id="deep", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _grid_block(x0: int, y0: int, w: int, h: int, streets_side: str, block_id: str) -> Block:
    """Minimal standalone copy of `test_region.py::_grid_block` (bottom/top sides only -- all
    this module's region fixture needs), kept local so `scoring_fixtures` doesn't import a test
    module: a w x h grid of unit parcels at (x0, y0), with street frontage on only one outer
    side, for building a deep block/region."""
    polys, ids = [], []
    for i in range(w):
        for j in range(h):
            polys.append(Polygon([
                (x0 + i, y0 + j), (x0 + i + 1, y0 + j),
                (x0 + i + 1, y0 + j + 1), (x0 + i, y0 + j + 1)]))
            ids.append(i * h + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    sides = {"bottom": LineString([(x0, y0), (x0 + w, y0)]),
            "top": LineString([(x0, y0 + h), (x0 + w, y0 + h)])}
    streets = gpd.GeoDataFrame(geometry=[sides[streets_side]], crs=UTM)
    return Block(block_id=block_id, crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _region_deep() -> Block:
    """The deep region fixture from `test_region.py`
    (`test_greedy_arterial_beats_dijkstra_directness_auc_on_a_deep_region`): two 3x6 blocks
    joined by `region_block`, frontage on opposite outer short ends, so the joined interior is
    deep/meshy -- deeper than the compact 10-parcel 1808 block."""
    a = _grid_block(0, 0, 3, 6, "bottom", "a")
    b = _grid_block(3, 0, 3, 6, "top", "b")
    return region_block([a, b])


def sampled_fixtures() -> list[tuple[str, Block, gpd.GeoDataFrame | None, dict[str, Any]]]:
    b1808 = _block_1808()
    fixtures = [(k, b1808, _roads(b1808, _REF, k), _REF[k])
               for k in ("no_roads", "dijkstra", "arterial_buildable")]

    coincident = _block_coincident()
    fixtures.append(("coincident", coincident, None, _REF_EXTRA["coincident"]))

    sparse_chord_block = _block_sparse_chord()
    chord = gpd.GeoDataFrame(geometry=[LineString([(1.5, 0.0), (1.5, 7.0)])], crs=UTM)
    fixtures.append(("sparse_chord", sparse_chord_block, chord, _REF_EXTRA["sparse_chord"]))

    region = _region_deep()
    for k in ("deep_region_dijkstra", "deep_region_arterial"):
        fixtures.append((k, region, _roads(region, _REF_EXTRA, k), _REF_EXTRA[k]))

    return fixtures
