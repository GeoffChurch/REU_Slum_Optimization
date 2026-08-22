"""Bake examples/screen-map/ -- the Cape Town and Nairobi city tiers for the ScreenMap widget.

The widget lets a reader pick one of four cheap screening metrics (design §3.1: `density`,
`depth_density_proxy`, `density_compactness`, `depth_proxy` -- all arithmetic on `building_count`,
area and perimeter, so the metric selector costs nothing client-side) and see which blocks it
selects at its shipped absolute floor, and -- for Cape Town, which has ground truth -- how well that
selection matches the City's own informal-structure survey (`reblock.data.informal`). Nairobi ships
the same map, the same floors and the same per-floor pool sizes, but no precision or recall:
`reblock.data.informal` records a searched-and-documented absence of an equivalent Kenyan layer, and
the whole point (design §3.4) is that an ABSOLUTE floor still transfers there where a percentile
re-defining "top X%" of a different corpus would not.

Outputs, into examples/screen-map/:

    capetown.json   the Cape Town bundle (16,451 blocks, ground truth included)
    nairobi.json    the Nairobi bundle (3,500 blocks, no `informal` field -- see its README)
    screen_map.png  fallback figure -- Cape Town at the shipped depth_density_proxy floor
    README.md       generated

The bake also writes web/src/screen_map.d.ts, generated from the same schema this file emits.

Reproduce with `pixi run python -m scripts.gen_screen_map`.
"""
from __future__ import annotations

import csv
import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict, cast

import geopandas as gpd
import matplotlib
import numpy as np
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from reblock.data.informal import label_blocks, settlement_extents
from reblock.data.provision import cached_kblock_source
from reblock.render import _CONTEXT_OUTLINE, _DISPLACED_PT, _PARCEL_LW, _ROAD_COLOR, save_render
from scripts._bundle_io import cm, polygon_rings, sigfig

log = logging.getLogger(__name__)

OUT = Path("examples/screen-map")
CITIES = {"capetown": 32734, "nairobi": 32737}
MIN_COUNT = 30
SIMPLIFY_M = 5.0     # design §1.1: 5.49 MB / 1.85 MB gz for Cape Town, sub-pixel at city zoom
BAKEOFF_CSV = Path("examples/screen-bakeoff/screen_comparison.csv")

# The site's spine: the one block every later stage is about. READ, never typed -- perm-graph,
# displacement-field and method-comparison all pin it, and region-grow seeds from it, so taking it
# from an artifact is what keeps the city map's marker and those four figures from drifting apart.
FOLLOW_SOURCE = Path("examples/perm-graph/bundle.json")
FOLLOW_CITY = "capetown"

DTS = Path("web/src/screen_map.d.ts")


class CityFloor(TypedDict):
    """One shipped absolute floor -- `web/src/screen_map.d.ts`'s `CityFloor`. `value` and (for
    Cape Town) `precision`/`recall` are READ from `BAKEOFF_CSV`, never typed (see
    `load_floor_specs`); `n` -- the pool size at that floor -- is recomputed per city from that
    city's OWN `n`/`area_m2`/`perimeter_m`, because Nairobi has no bake-off row to read it from and
    the design's whole point (§3.4) is that the floor transfers to a corpus with no ground truth."""
    metric: str
    value: float
    n: int
    precision: float | None
    recall: float | None


class CityFollow(TypedDict):
    """Where to mark the followed block -- `web/src/screen_map.d.ts`'s `CityFollow`. `index` is
    that block's position in the column arrays, so the widget reads its row without searching
    `block_id`; `x`/`y` are origin-relative like every ring coordinate."""
    block_id: str
    index: int
    x: float
    y: float


class CityEncoding(TypedDict):
    """`Encoding`'s JSON shape -- `web/src/screen_map.d.ts`'s `CityEncoding`."""
    base_color: str
    selected_color: str
    informal_color: str
    follow_color: str
    block_lw: float
    pad: float


class CityBundle(TypedDict):
    """The whole artifact -- `web/src/screen_map.d.ts`'s `CityBundle`, generated from this file.

    Columnar, not an array of per-block objects: see `web/src/screen_map.d.ts`'s own comment on
    `rings` for why (16,451 blocks with repeated JSON keys would add megabytes of field names).
    `informal` is `NotRequired`: Cape Town carries it, Nairobi OMITS it entirely -- a missing field
    is a type the widget must handle; a null column is a field that looks answerable and is not.
    `follow` takes the same shape for the same reason: the followed block is in Cape Town, and
    Nairobi has no answer to give.
    """
    city: str
    crs_epsg: int
    origin: list[float]
    n_blocks: int
    block_id: list[str]
    n: list[int]
    area_m2: list[float]
    perimeter_m: list[float]
    rings: list[list[list[list[float]]]]
    informal: NotRequired[list[int]]
    follow: NotRequired[CityFollow]
    floors: list[CityFloor]
    encoding: CityEncoding


DTS_TEMPLATE = """// GENERATED by scripts/gen_screen_map.py -- do not edit.
// Regenerate: pixi run python -m scripts.gen_screen_map
export interface CityFloor {
  /** The metric this floor belongs to, as `reblock.metric` names it. */
  metric: string;
  value: number;
  n: number;
  /** Read from examples/screen-bakeoff/screen_comparison.csv, which computes them independently
   * of this bundle -- so the widget's own prefix arithmetic has something to be checked against. */
  precision: number | null;
  recall: number | null;
}
export interface CityFollow {
  /** The one block the whole site follows: perm-graph, displacement-field and method-comparison
   * pin this `block_id` and region-grow seeds from it. `index` is its position in the column
   * arrays below, so the widget reads its row without searching `block_id`. */
  block_id: string;
  index: number;
  /** Origin-relative like every ring coordinate, and a `representative_point()` -- inside the
   * polygon even where the block is concave or holed. A POINT, not an outline: the median block is
   * ~0.6 CSS px² at the shipped canvas size (see render/city.ts), so the marker has to be a ring
   * of fixed SCREEN size around this rather than the block's own boundary. */
  x: number;
  y: number;
}
export interface CityEncoding {
  base_color: string;
  selected_color: string;
  informal_color: string;
  /** The follow marker. A blue, off the grey/red/gold axis the other three encode meaning along.
   * Carried by BOTH cities even though only Cape Town carries `follow`, so a city switch cannot
   * leave the colour undefined. */
  follow_color: string;
  block_lw: number;
  pad: number;
}
export interface CityBundle {
  city: string;
  crs_epsg: number;
  origin: [number, number];
  n_blocks: number;
  /** Column arrays, not an array of objects: 16,451 blocks with repeated keys would add megabytes
   * of field names. Same shape as field.json's `buildings: {x, y, r}`. */
  block_id: string[];
  n: number[];
  area_m2: number[];
  perimeter_m: number[];
  /** Per block, exterior ring first then interiors. Fill even-odd. */
  rings: [number, number][][][];
  /** 0/1 ground truth. ABSENT for Nairobi -- see the README. Not a null column: a null column is
   * a field that looks answerable and is not. */
  informal?: number[];
  /** ABSENT for Nairobi, like `informal` and for the same reason: the followed block is in Cape
   * Town, and a null field is one that looks answerable and is not. */
  follow?: CityFollow;
  floors: CityFloor[];
  encoding: CityEncoding;
}
"""


@dataclass(frozen=True)
class Encoding:
    """What `screen_map.png` draws with AND what `encoding` in each bundle carries -- one source
    for both (see `gen_region_grow.Encoding`'s docstring for why that matters: a bundle and the PNG
    beside it drawing from two separately-typed numbers is how a JS-on/JS-off divergence like D2's
    `street_lw` mismatch happens).

    `base_color`/`block_lw` reuse `_CONTEXT_OUTLINE`/`_PARCEL_LW`, the same pale thin wireframe role
    RegionGrow's `hood_color`/`hood_lw` draws every unselected unit with. `selected_color` reuses
    `_DISPLACED_PT`, the same emphasis red RegionGrow's `region_color` draws a chosen region with --
    here, the floor's own selection. `informal_color` has no `render.py` analogue (no other bundle
    here ships ground truth), so it reuses `gen_screen_bakeoff.py`'s own gold
    (`plot_city`'s `edgecolor="#d98c00"`), the established colour for "real settlement, City survey"
    in this exact problem domain. `pad` matches RegionGrow's and DisplacementField's own widget-side
    framing margin; like theirs, it has no PNG equivalent, so `screen_map.png` below does not
    consume it.

    `follow_color` reuses `_ROAD_COLOR`, the one blue in `render.py`'s palette. What separates it
    here is HUE, not lightness: the three colours already in this encoding are a pale grey, a red
    and a gold, and the palette's near-blacks (`_BOUNDARY_COLOR`, `_OWN_PT`) are legible against
    those too but read as one more outline on a map whose outlines are all dark. A blue ring is the
    only mark here that is off the grey/red/gold axis the rest of the figure encodes meaning along,
    which is what a marker findable among 16,451 blocks needs.
    """
    base_color: str
    selected_color: str
    informal_color: str
    follow_color: str
    block_lw: float
    pad: float


ENCODING = Encoding(base_color=_CONTEXT_OUTLINE, selected_color=_DISPLACED_PT,
                    informal_color="#d98c00", follow_color=_ROAD_COLOR, block_lw=_PARCEL_LW,
                    pad=0.04)
ENCODING_DICT = CityEncoding(base_color=ENCODING.base_color, selected_color=ENCODING.selected_color,
                             informal_color=ENCODING.informal_color,
                             follow_color=ENCODING.follow_color, block_lw=ENCODING.block_lw,
                             pad=ENCODING.pad)

# gen_screen_bakeoff.py's own METRICS display strings (screen_comparison.csv's `metric` column) ->
# this bundle's `reblock.metric`-style names, for the two rows that carry an absolute floor
# (DEPTH_DENSITY_PROXY_FLOOR, DENSITY_COMPACTNESS_FLOOR). Closed and exhaustive over what the CSV
# ships today: a floor row for a metric not in this dict raises rather than being silently dropped
# from the bundle.
FLOOR_METRIC_NAMES = {
    "depth_density proxy   √(nA)/P · n/A": "depth_density_proxy",
    "density_compactness   n/P²": "density_compactness",
}


class _FloorSpec(NamedTuple):
    """One row of `load_floor_specs`' output: Cape Town's own calibration, read from the CSV."""
    metric: str
    value: float
    csv_n: int
    precision: float
    recall: float


def load_floor_specs(csv_path: Path) -> list[_FloorSpec]:
    """Every `BAKEOFF_CSV` row that carries a shipped absolute floor, translated to this bundle's
    metric names via `FLOOR_METRIC_NAMES`. `value`/`precision`/`recall` are READ from the CSV, not
    typed -- `gen_screen_bakeoff.py` computes them independently, which is what makes them a real
    check on this bundle rather than a restatement of it."""
    specs: list[_FloorSpec] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row["floor"]:
                continue
            if row["metric"] not in FLOOR_METRIC_NAMES:
                raise ValueError(
                    f"{csv_path}: floor row for unrecognized metric {row['metric']!r} -- add it "
                    f"to FLOOR_METRIC_NAMES, or the shipped floor set changed under this bake")
            specs.append(_FloorSpec(
                metric=FLOOR_METRIC_NAMES[row["metric"]], value=float(row["floor"]),
                csv_n=int(float(row["floor_n"])), precision=float(row["floor_prec"]),
                recall=float(row["floor_recall"])))
    if not specs:
        raise ValueError(f"{csv_path}: no floor rows found -- the bake-off CSV is stale or empty")
    return specs


def _score(name: str, n: NDArray[np.float64], a: NDArray[np.float64],
          p: NDArray[np.float64]) -> NDArray[np.float64]:
    """The four cheap screens (design §3.1), vectorized, closed over the same set the widget's own
    client-side formulas and `tests/test_screen_map_bundle.py`'s `_metric` cover -- an unrecognized
    name raises rather than silently returning zeros for every block."""
    if name == "depth_density_proxy":
        return cast(NDArray[np.float64], np.sqrt(n * a) / p * (n / a))
    if name == "density":
        return cast(NDArray[np.float64], n / a)
    if name == "density_compactness":
        return cast(NDArray[np.float64], n / p ** 2)
    if name == "depth_proxy":
        return cast(NDArray[np.float64], np.sqrt(n * a) / p)
    raise ValueError(name)


def load_blocks(city: str, epsg: int) -> gpd.GeoDataFrame:
    """MIN_COUNT-filtered blocks, reprojected to `epsg` -- the same filter and CRS
    `gen_screen_bakeoff.py`'s own `load()` uses for Cape Town, generalised to both cities. The
    zero-area/zero-perimeter guard is defensive (measured empirically: no block above MIN_COUNT is
    degenerate on either cached parquet, so it never fires today) and mirrors that script's own."""
    src = cached_kblock_source(city, min_buildings=MIN_COUNT)
    raw = gpd.read_parquet(src.blocks_path, columns=["block_id", "building_count", "geometry"])
    raw["block_id"] = raw["block_id"].astype(str)
    b = raw.to_crs(epsg)
    area = b.geometry.area.to_numpy()
    perim = b.geometry.length.to_numpy()
    count = b["building_count"].to_numpy()
    mask = (count >= MIN_COUNT) & (area > 0) & (perim > 0)
    return cast(gpd.GeoDataFrame, b[mask].reset_index(drop=True))


def build_bundle(city: str, epsg: int, blocks: gpd.GeoDataFrame, simplified: list[BaseGeometry],
                 floor_specs: list[_FloorSpec]) -> CityBundle:
    """The bundle for one city. `simplified` is `blocks.geometry.simplify(SIMPLIFY_M)`, computed
    once by the caller and reused for `screen_map.png` -- a second simplify pass over 16,451
    polygons would just be wasted work."""
    minx, miny, _, _ = blocks.total_bounds
    ox, oy = float(minx), float(miny)
    block_ids = [str(x) for x in blocks["block_id"]]
    counts = [int(x) for x in blocks["building_count"]]
    areas = [sigfig(float(x)) for x in blocks.geometry.area.to_numpy()]
    perims = [sigfig(float(x)) for x in blocks.geometry.length.to_numpy()]
    rings = [polygon_rings(g, ox, oy, what=f"{city} block {bid!r}")
            for bid, g in zip(block_ids, simplified, strict=True)]

    follow: CityFollow | None = None
    if city == FOLLOW_CITY:
        want = json.loads(FOLLOW_SOURCE.read_text(encoding="utf-8"))["block_id"]
        if want not in block_ids:
            raise ValueError(
                f"{FOLLOW_SOURCE} pins block {want!r}, which is not among {city}'s "
                f"{len(block_ids)} blocks above MIN_COUNT={MIN_COUNT} -- the site's spine and this "
                f"bundle disagree about which block the walkthrough follows")
        idx = block_ids.index(want)
        # representative_point(), not centroid: guaranteed INSIDE the polygon even where the block
        # is concave or holed, so the marker can never be drawn over a neighbour.
        pt = simplified[idx].representative_point()
        follow = CityFollow(block_id=want, index=idx,
                            x=cm(float(pt.x) - ox), y=cm(float(pt.y) - oy))

    n_arr = np.asarray(counts, dtype=np.float64)
    a_arr = np.asarray(areas, dtype=np.float64)
    p_arr = np.asarray(perims, dtype=np.float64)
    floors: list[CityFloor] = []
    for spec in floor_specs:
        pool = int((_score(spec.metric, n_arr, a_arr, p_arr) >= spec.value).sum())
        if city == "capetown" and pool != spec.csv_n:
            raise ValueError(
                f"{spec.metric}: recomputed pool {pool} != bake-off CSV floor_n {spec.csv_n} -- "
                f"the bundle's n/area_m2/perimeter_m no longer reproduce the published floor")
        floors.append(CityFloor(
            metric=spec.metric, value=sigfig(spec.value), n=pool,
            precision=sigfig(spec.precision) if city == "capetown" else None,
            recall=sigfig(spec.recall) if city == "capetown" else None))

    bundle = CityBundle(
        city=city, crs_epsg=epsg, origin=[cm(ox), cm(oy)], n_blocks=len(blocks),
        block_id=block_ids, n=counts, area_m2=areas, perimeter_m=perims, rings=rings,
        floors=floors, encoding=ENCODING_DICT)
    if city == "capetown":
        extents = settlement_extents(city, epsg=epsg)
        _, label = label_blocks(blocks, extents)
        bundle["informal"] = [int(x) for x in label]
    if follow is not None:
        bundle["follow"] = follow
    return bundle


def _render_screen_map(gdf: gpd.GeoDataFrame, selected_ids: set[str], informal_ids: set[str],
                       follow_xy: tuple[float, float]) -> Figure:
    """The fallback figure: Cape Town at the shipped `depth_density_proxy` floor. Every block's
    thin wireframe in `base_color`, the City's own ground truth filled in `informal_color`, and the
    floor's own selection outlined in `selected_color` on top -- so a gold fill with a red outline
    reads as a hit, a bare red outline reads as a false positive, and a gold fill with no outline
    reads as a miss.

    `follow_xy` is the followed block's point in `gdf`'s own CRS -- `gdf.plot` draws in world
    coordinates, so this is NOT the bundle's origin-relative pair."""
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, facecolor="none", edgecolor=ENCODING.base_color, linewidth=ENCODING.block_lw,
             zorder=1)
    informal = cast(gpd.GeoDataFrame, gdf[gdf["block_id"].isin(informal_ids)])
    informal.plot(ax=ax, facecolor=ENCODING.informal_color, edgecolor="none", zorder=2)
    selected = cast(gpd.GeoDataFrame, gdf[gdf["block_id"].isin(selected_ids)])
    selected.plot(ax=ax, facecolor="none", edgecolor=ENCODING.selected_color,
                 linewidth=ENCODING.block_lw * 2, zorder=3)
    # The follow marker. Colour from `ENCODING`, which is also what `encoding` carries into both
    # bundles, and centred on the same block the bundle's `follow` names -- one constant and one
    # block feeding this fallback and the widget, so a JS-off or print reader sees the block the
    # prose says the page follows and the two cannot drift apart independently.
    #
    # A fixed-size ring rather than the block's own outline, for the reason the widget has: at city
    # zoom a block is sub-pixel (`web/src/render/city.ts` measures the median at ~0.6 CSS px² on a
    # 700 px canvas), so an outline of it would draw nothing findable. `markersize` is a diameter
    # in POINTS, set against the figure rather than the map's extent; 12.3 pt is a larger fraction
    # of this figure's width than the widget's 6 CSS px radius is of its canvas, deliberately,
    # because the README displays this PNG well below its baked pixel width.
    ax.plot(*follow_xy, marker="o", markersize=12.3, markerfacecolor="none",
            markeredgecolor=ENCODING.follow_color, markeredgewidth=2.0, zorder=4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0)
    return fig


def readme_markdown(bundles: dict[str, CityBundle], sizes: dict[str, tuple[int, int]]) -> str:
    """This directory's README, written from the bundles it documents -- see
    `gen_displacement_field.readme_markdown`'s docstring for why generated beats handwritten: every
    fact worth stating here is already in `capetown.json`/`nairobi.json`, and a handwritten copy of
    a number is a copy that rots."""
    ct, nb = bundles["capetown"], bundles["nairobi"]
    ct_holes = sum(len(r) - 1 for r in ct["rings"])
    nb_holes = sum(len(r) - 1 for r in nb["rings"])
    ct_json, ct_gz = sizes["capetown"]
    nb_json, nb_gz = sizes["nairobi"]
    ct_informal = ct.get("informal")
    assert ct_informal is not None, "capetown must carry ground truth"
    n_informal = sum(ct_informal)
    ct_follow = ct.get("follow")
    assert ct_follow is not None, "capetown must carry the followed block"

    def floor_row(f: CityFloor) -> str:
        prec = "—" if f["precision"] is None else f"{f['precision']:.1%}"
        rec = "—" if f["recall"] is None else f"{f['recall']:.1%}"
        return f"| `{f['metric']}` | {f['value']:g} | {f['n']:,} | {prec} | {rec} |"

    ct_rows = "\n".join(floor_row(f) for f in ct["floors"])
    nb_rows = "\n".join(floor_row(f) for f in nb["floors"])

    return f"""<!-- GENERATED by scripts/gen_screen_map.py -- do not edit. Regenerate:
     pixi run python -m scripts.gen_screen_map -->

# The ScreenMap city tiers

The figure set for the site's ScreenMap widget: pick one of four cheap screening metrics, see which
blocks it selects at its shipped absolute floor, and -- for Cape Town -- how well that selection
matches the City of Cape Town's own informal-structure survey
({n_informal:,} of {ct['n_blocks']:,} blocks are really informal by that survey,
via `reblock.data.informal`).

![Cape Town at the shipped depth_density_proxy floor: gold = real informal settlement, red outline \
= selected, blue ring = the block the rest of the site follows](screen_map.png)

The blue ring marks `{ct_follow['block_id']}` (index {ct_follow['index']:,} of the columns below)
-- the single block every later stage of the site is about: perm-graph, displacement-field and
method-comparison all pin it, and region-grow seeds from it. Cape Town's bundle carries it as
`follow`; Nairobi omits the field, the same way it omits `informal`. It is a POINT, not an outline,
because at this zoom a block covers well under one pixel.

`capetown.json` and `nairobi.json` are the payloads the widget fetches: every block's
`building_count`, area, perimeter and simplified boundary rings (fill even-odd), plus the shipped
floors and a rendering `encoding`. Metrics themselves are **not** shipped as columns -- they are
arithmetic on `building_count`/area/perimeter cheap enough for the widget to compute client-side the
same way `reblock.metric` computes them (design §3.1), so switching metrics costs one client-side
sort, not a re-fetch. The bake also writes `web/src/screen_map.d.ts`, which is what makes a renamed
field a TypeScript error rather than a blank map.

| city | blocks | interior rings | JSON (5 m simplify, compact) | gzip |
|---|---|---|---|---|
| Cape Town | {ct['n_blocks']:,} | {ct_holes:,} | {ct_json / 1e6:.2f} MB | {ct_gz / 1e6:.2f} MB |
| Nairobi | {nb['n_blocks']:,} | {nb_holes:,} | {nb_json / 1e6:.2f} MB | {nb_gz / 1e6:.2f} MB |

**Shipped floors.** `n` is each floor's pool size on THAT city's own blocks -- for Nairobi this is
recomputed directly (there is no bake-off row to read it from), not copied from Cape Town's. Cape
Town's `precision`/`recall` are read from `examples/screen-bakeoff/screen_comparison.csv`, which
`gen_screen_bakeoff.py` computes independently of this bundle -- two paths agreeing is the guard
`tests/test_screen_map_bundle.py` checks.

Cape Town:

| metric | floor | pool | precision | recall |
|---|---|---|---|---|
{ct_rows}

Nairobi -- same floors, same formulas, **no ground truth to check them against**:

| metric | floor | pool | precision | recall |
|---|---|---|---|---|
{nb_rows}

**Nairobi omits `informal` rather than shipping nulls.** No equivalent published informal-settlement
layer was found for Nairobi -- searched and documented in `reblock.data.informal` (also recorded in
`examples/screen-bakeoff/README.md`'s own caveats). A missing field is a type the widget must
handle explicitly; a null column is a field that looks answerable and is not. That the SAME absolute
floors still produce a plausible, nonzero pool on a corpus with no calibration is itself the point
(design §3.4): an absolute threshold transfers across corpora where a percentile does not, because a
percentile silently redefines "top X%" every time the corpus changes size.

Regenerate: `pixi run python -m scripts.gen_screen_map`
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)

    floor_specs = load_floor_specs(BAKEOFF_CSV)
    log.info("floors: %s", [(s.metric, s.value) for s in floor_specs])

    bundles: dict[str, CityBundle] = {}
    sizes: dict[str, tuple[int, int]] = {}
    render_gdf: gpd.GeoDataFrame | None = None
    follow_xy: tuple[float, float] | None = None

    for city, epsg in CITIES.items():
        blocks = load_blocks(city, epsg)
        log.info("%s: %d blocks above MIN_COUNT=%d", city, len(blocks), MIN_COUNT)

        simplified = list(blocks.geometry.simplify(SIMPLIFY_M))
        bundle = build_bundle(city, epsg, blocks, simplified, floor_specs)
        bundles[city] = bundle

        js = json.dumps(bundle, separators=(",", ":")) + "\n"
        out_path = OUT / f"{city}.json"
        out_path.write_text(js, encoding="utf-8")
        gz_size = len(gzip.compress(js.encode("utf-8"), compresslevel=9))
        sizes[city] = (len(js.encode("utf-8")), gz_size)
        log.info("wrote %s (%.2f MB, %.2f MB gz)", out_path, len(js) / 1e6, gz_size / 1e6)

        if city == "capetown":
            render_gdf = gpd.GeoDataFrame({"block_id": bundle["block_id"]},
                                          geometry=simplified, crs=blocks.crs)
        if city == FOLLOW_CITY:
            follow = bundle.get("follow")
            assert follow is not None, f"build_bundle must give {FOLLOW_CITY} a `follow`"
            # World coordinates for the figure, taken from `simplified` at the BUNDLE's own index:
            # the PNG's ring and the widget's ring are then the same point, converted twice, rather
            # than two lookups that could pick different blocks.
            world = simplified[follow["index"]].representative_point()
            follow_xy = (float(world.x), float(world.y))

    assert render_gdf is not None, "capetown must be baked to render screen_map.png"
    assert follow_xy is not None, f"FOLLOW_CITY={FOLLOW_CITY!r} must be one of CITIES"
    ct = bundles["capetown"]
    shipped = {f["metric"]: f for f in ct["floors"]}["depth_density_proxy"]
    n_arr = np.asarray(ct["n"], dtype=np.float64)
    a_arr = np.asarray(ct["area_m2"], dtype=np.float64)
    p_arr = np.asarray(ct["perimeter_m"], dtype=np.float64)
    selected_mask = _score("depth_density_proxy", n_arr, a_arr, p_arr) >= shipped["value"]
    ct_informal = ct.get("informal")
    assert ct_informal is not None, "capetown must carry ground truth"
    selected_ids = {ct["block_id"][i] for i in range(ct["n_blocks"]) if selected_mask[i]}
    informal_ids = {ct["block_id"][i] for i, v in enumerate(ct_informal) if v}

    fig = _render_screen_map(render_gdf, selected_ids, informal_ids, follow_xy)
    save_render(fig, OUT / "screen_map.png")
    plt.close(fig)
    log.info("wrote %s", OUT / "screen_map.png")

    DTS.parent.mkdir(parents=True, exist_ok=True)
    DTS.write_text(DTS_TEMPLATE, encoding="utf-8")
    log.info("wrote %s", DTS)

    (OUT / "README.md").write_text(readme_markdown(bundles, sizes), encoding="utf-8")
    log.info("wrote %s", OUT / "README.md")


if __name__ == "__main__":
    main()
