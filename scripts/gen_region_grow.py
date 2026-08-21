"""Bake examples/region-grow/ -- the RegionGrow widget's neighbourhood bundle.

The widget runs the PRODUCTION greedy in the browser (web/src/model/accretion.ts), so this bundle
ships the raw quantities that greedy needs -- building_count, area, perimeter and a precomputed
adjacency list -- rather than a baked animation. What it also ships is `reference`: the accretion
`DenseClusterRegionBuilder` itself produces for several seeds and budgets, which is what pins the
TypeScript to this code instead of to a re-implementation of it.

The seed is ZAF.9.3.1_1_40972, the same block PermGraph, Frontier and DisplacementField pin.

Outputs, into examples/region-grow/:

    hood.json    the bundle (schema: web/src/hood.d.ts, generated here)
    hood.png     the fallback figure, drawn from the same encoding the bundle carries
    README.md    generated

Reproduce with `pixi run python -m scripts.gen_region_grow`.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from reblock.data.provision import cached_kblock_source
from reblock.region import DenseClusterRegionBuilder, _block_adjacency, _projected
from reblock.render import (
    _BOUNDARY_COLOR,
    _BOUNDARY_LW,
    _CONTEXT_OUTLINE,
    _DISPLACED_PT,
    _PARCEL_LW,
    save_render,
)
from scripts._bundle_io import polygon_rings, sigfig

log = logging.getLogger(__name__)

OUT = Path("examples/region-grow")
DTS = Path("web/src/hood.d.ts")


class HoodBlock(TypedDict):
    """One neighbourhood block, in the shape the browser's greedy needs directly -- no server
    round trip, no re-derivation. `n`/`area_m2`/`perimeter_m` are measured on RAW, un-simplified
    geometry -- the SAME basis `DenseClusterRegionBuilder._depth_proxy` scores against internally
    (`region.py`'s `metric = _projected(block_geoms)` is never simplified) -- so the widget's own
    greedy reproduces production's tie-breaks at every budget the slider can reach, not only the
    12 pinned `reference` cases below. Only `rings` (what gets drawn) comes from the simplified
    geometry; a rendering-only simplification must not also change which block wins a tie."""
    block_id: str
    n: int
    area_m2: float
    perimeter_m: float
    rings: list[list[list[float]]]
    adj: list[int]


class GrowthCase(TypedDict):
    """One seed/budget pair, and `DenseClusterRegionBuilder`'s OWN accretion order for it -- the
    parity fixture `tests/test_region_grow_bundle.py::test_bundle_is_what_production_builds_today`
    recomputes against live Python, and what the TypeScript greedy is held to."""
    seed: str
    max_buildings: int
    order: list[str]
    buildings: int


@dataclass(frozen=True)
class Budget:
    """The slider's own range, in building_count -- the same unit growth is budgeted in.

    `min = 150` is deliberate, not an arbitrary floor: at that value the region is the seed alone,
    the design's §1.3 finding SHOWN rather than hidden
    (`test_the_shipped_budget_floor_is_a_no_op_on_the_seed`). `default = 3000` is what every
    `conf/example/*.yaml` actually sets, so the widget boots on a budget the rest of the site
    already uses."""
    min: int
    max: int
    step: int
    default: int


class BudgetDict(TypedDict):
    """`Budget`'s JSON shape -- `web/src/hood.d.ts`'s `Budget` interface."""
    min: int
    max: int
    step: int
    default: int


@dataclass(frozen=True)
class Encoding:
    """What `hood.png` draws with AND what `encoding` in the bundle carries -- one source for
    both, not two lists kept in step by hand. D2 shipped `street_lw: 1.0` in a bundle while the PNG
    it sat beside drew `1.3`, a JS-on/JS-off divergence no test caught; a dataclass whose fields
    feed both the matplotlib call and the JSON field cannot drift the same way. `region_alpha` is
    the same fix applied to itself: it used to be a bare `alpha=0.55` inside `_render_hood`'s own
    `region.plot(...)` call, with no field for the widget to read, so the widget drew the region
    fill at full opacity while the PNG beside it drew translucent -- correct by the schema, wrong
    on the page, precisely the class of bug this docstring already warns about.

    No named `render.py` constant serves this exact combination (neighbourhood context + a grown
    region's fill + its frontier + the seed) the way `render_field` serves DisplacementField, so
    only the roles that already match an existing one reuse it: `hood_color`/`hood_lw` are the same
    pale context outline `_CONTEXT_OUTLINE`/`_PARCEL_LW` draw elsewhere, `region_color` the same
    emphasis red `_DISPLACED_PT` draws elsewhere, `seed_color` the same near-black `_BOUNDARY_COLOR`
    an important outline draws elsewhere. `frontier_color` has no existing analogue (no other figure
    here draws "candidate, not yet chosen"); it and the seed outline both use `region_lw` for their
    stroke weight rather than adding fields the schema (`web/src/hood.d.ts`'s `HoodEncoding`) does
    not declare."""
    hood_color: str
    hood_lw: float
    region_color: str
    region_lw: float
    region_alpha: float
    seed_color: str
    frontier_color: str
    pad: float


class EncodingDict(TypedDict):
    """`Encoding`'s JSON shape -- `web/src/hood.d.ts`'s `HoodEncoding` interface."""
    hood_color: str
    hood_lw: float
    region_color: str
    region_lw: float
    region_alpha: float
    seed_color: str
    frontier_color: str
    pad: float


class HoodBundle(TypedDict):
    """The whole artifact -- `web/src/hood.d.ts`'s `HoodBundle`, generated from this file."""
    city: str
    seed: str
    origin: list[float]
    crs_epsg: int
    blocks: list[HoodBlock]
    budget: BudgetDict
    encoding: EncodingDict
    reference: list[GrowthCase]


CITY = "capetown"
SEED = "ZAF.9.3.1_1_40972"
MIN_COUNT = 30            # the same filter gen_screen_bakeoff.py applies
HOPS = 7                  # MEASURED: the budget-10,000 accretion reaches 7 hops; 5 leaves 2 out
SIMPLIFY_M = 1.0          # region scale: 5 m would be visible here (design §1.2)

BUDGET = Budget(min=150, max=10_000, step=50, default=3000)
REFERENCE_BUDGETS = (150, 600, 3000, 10_000)
# SEED plus two neighbours whose accretion order differs from sorted order (measured, Step 4 of
# the task brief): both must give an order at least 4 blocks long that is NOT already sorted, or a
# downstream test comparing accretion order to `sorted()` cannot tell the two apart and guards
# nothing (D2's defect #7 -- a fixture satisfied by its own twin). MEASURED by growing every one of
# SEED's 12 nearest neighbours (hops=2) at budget=3000: all 12 qualified (orders 9-12 blocks long,
# all != sorted), so no need to widen the scan -- see task-4-report.md for the full scan.
REFERENCE_SEEDS = (SEED, "ZAF.9.3.1_1_40973", "ZAF.9.3.1_1_40144")

# hood/region reuse render.py's own named roles; frontier has no analogue there (see Encoding's
# docstring). `pad` matches DisplacementField's ENCODING.pad -- the same widget-side breathing room
# around a fitted extent; like that bundle's `pad`, it has no PNG equivalent (render_field's own
# framing is not pinned by it either), so `hood.png` below does not consume it.
ENCODING = Encoding(
    hood_color=_CONTEXT_OUTLINE, hood_lw=_PARCEL_LW,
    region_color=_DISPLACED_PT, region_lw=_BOUNDARY_LW, region_alpha=0.55,
    seed_color=_BOUNDARY_COLOR,
    frontier_color="#f5a623",
    pad=0.04,
)

DTS_TEMPLATE = """// GENERATED by scripts/gen_region_grow.py -- do not edit.
// Regenerate: pixi run python -m scripts.gen_region_grow
// This file is what makes a renamed Python field a TypeScript error instead of a blank panel.
export interface HoodBlock {
  block_id: string;
  /** building_count. The growth budget is measured in these. */
  n: number;
  area_m2: number;
  perimeter_m: number;
  /** Exterior ring first, then interiors. 7 of the 213 blocks have one. Fill even-odd. */
  rings: [number, number][][];
  /** Indices into `blocks`, not block_ids -- the greedy runs over these directly. */
  adj: number[];
}
/** Production's own accretion, for one seed at one budget -- the parity fixtures. */
export interface GrowthCase {
  seed: string;
  max_buildings: number;
  /** block_ids in ACCRETION order, straight out of DenseClusterRegionBuilder. */
  order: string[];
  buildings: number;
}
export interface Budget { min: number; max: number; step: number; default: number }
export interface HoodEncoding {
  hood_color: string;
  hood_lw: number;
  region_color: string;
  region_lw: number;
  /** The region fill's opacity -- matches hood.png's own alpha, so JS-on and JS-off draw
   * the same fill. */
  region_alpha: number;
  seed_color: string;
  frontier_color: string;
  pad: number;
}
export interface HoodBundle {
  city: string;
  /** The pinned seed -- the same block PermGraph, Frontier and DisplacementField use. */
  seed: string;
  /** UTM easting/northing subtracted from every coordinate; all geometry is local metres. */
  origin: [number, number];
  crs_epsg: number;
  blocks: HoodBlock[];
  budget: Budget;
  encoding: HoodEncoding;
  reference: GrowthCase[];
}
"""


def load_blocks(city: str) -> gpd.GeoDataFrame:
    """Blocks above MIN_COUNT, PROJECTED to the city UTM.

    Projection is not cosmetic. `_block_adjacency` measures `dwithin(STREET_TOL)` with
    STREET_TOL = 0.5, which is 0.5 m here and about 55 km in the parquet's native lon/lat -- where
    every block in the metro reads as adjacent to every other. Task 1 made the builders project
    defensively; this projects at the boundary so the requirement is visible at the call site.
    """
    src = cached_kblock_source(city, min_buildings=MIN_COUNT)
    geoms = _projected(src.block_geometries())
    above = cast(gpd.GeoDataFrame, geoms[geoms["building_count"] >= MIN_COUNT])
    return above.reset_index(drop=True)


def neighbourhood(blocks: gpd.GeoDataFrame, seed: str, *, hops: int) -> list[str]:
    """block_ids within `hops` block-adjacency steps of `seed`, sorted. Includes the seed."""
    ids = cast(list[str], list(blocks["block_id"]))
    idx = {b: i for i, b in enumerate(ids)}
    if seed not in idx:
        raise ValueError(f"seed {seed!r} is not among this frame's {len(ids)} block_ids")
    adj = _block_adjacency(list(blocks.geometry))
    start = idx[seed]
    seen = {start}
    depth = {start: 0}
    frontier: deque[int] = deque([start])
    while frontier:
        i = frontier.popleft()
        if depth[i] == hops:
            continue
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                depth[j] = depth[i] + 1
                frontier.append(j)
    return sorted(ids[i] for i in seen)


def growth(blocks: gpd.GeoDataFrame, seed: str, budget: int) -> GrowthCase:
    """One reference case, by calling DenseClusterRegionBuilder itself."""
    order = DenseClusterRegionBuilder(max_buildings=budget).build(blocks, [[seed]])[0]
    counts = dict(zip(cast(list[str], list(blocks["block_id"])), blocks["building_count"],
                      strict=True))
    buildings = int(sum(float(counts[b]) for b in order))
    return GrowthCase(seed=seed, max_buildings=budget, order=order, buildings=buildings)


def _render_hood(hood_gdf: gpd.GeoDataFrame, region_ids: set[str], frontier_ids: set[str],
                 seed: str) -> Figure:
    """The fallback figure: every neighbourhood block outlined in `hood_color`, the BOOT-budget
    (`BUDGET.default`) region filled in `region_color`, that region's frontier at the same budget
    outlined in `frontier_color`, and the seed picked out in `seed_color` -- the four layers
    `encoding` names, so a reader with JS off sees the same picture the widget boots with."""
    fig, ax = plt.subplots(figsize=(10, 10))
    hood_gdf.plot(ax=ax, facecolor="none", edgecolor=ENCODING.hood_color,
                 linewidth=ENCODING.hood_lw, zorder=1)
    region = cast(gpd.GeoDataFrame, hood_gdf[hood_gdf["block_id"].isin(region_ids)])
    region.plot(ax=ax, facecolor=ENCODING.region_color, edgecolor=ENCODING.region_color,
               linewidth=ENCODING.region_lw, alpha=ENCODING.region_alpha, zorder=2)
    frontier = cast(gpd.GeoDataFrame, hood_gdf[hood_gdf["block_id"].isin(frontier_ids)])
    if not frontier.empty:
        frontier.plot(ax=ax, facecolor="none", edgecolor=ENCODING.frontier_color,
                     linewidth=ENCODING.region_lw, zorder=3)
    seed_gdf = cast(gpd.GeoDataFrame, hood_gdf[hood_gdf["block_id"] == seed])
    seed_gdf.plot(ax=ax, facecolor="none", edgecolor=ENCODING.seed_color,
                 linewidth=ENCODING.region_lw, zorder=4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0)
    return fig


def readme_markdown(bundle: HoodBundle) -> str:
    """This directory's README, written from the bundle it documents -- see
    `gen_displacement_field.readme_markdown`'s docstring for why generated beats handwritten: every
    fact worth stating here is already in `hood.json`, and a handwritten copy of a number is a copy
    that rots (this repo has the specimen: `examples/nairobi/README.md` once claimed 89 blocks
    while every `meta.json` beside it said 43)."""
    holed = sum(1 for b in bundle["blocks"] if len(b["rings"]) > 1)
    budget = bundle["budget"]

    def row(c: GrowthCase) -> str:
        return (f"| `{c['seed']}` | {c['max_buildings']:,} | {len(c['order'])} | "
                f"{c['buildings']:,} |")

    rows = "\n".join(row(c) for c in bundle["reference"])
    return f"""<!-- GENERATED by scripts/gen_region_grow.py -- do not edit. Regenerate:
     pixi run python -m scripts.gen_region_grow -->

# The RegionGrow neighbourhood

The figure set for the site's RegionGrow widget: from one pinned seed block, grow a contiguous
region by block adjacency up to a buildings budget -- the same greedy `DenseClusterRegionBuilder`
runs server-side, replayed live in the browser.

![the pinned seed's neighbourhood, and the region grown from it at the slider's default budget]\
(hood.png)

`hood.png` is the **boot state**: the whole shipped neighbourhood outlined, the region grown from
the seed at the slider's default budget ({budget['default']:,} buildings) filled, that region's
frontier at the same budget outlined, and the seed itself picked out. `hood.json` is the payload
the widget fetches -- every neighbourhood block's `building_count`, area, perimeter and a
precomputed adjacency list, so the widget's own greedy needs no server round trip for any budget on
the slider. The bake also writes `web/src/hood.d.ts`, which is what makes a renamed field a
TypeScript error rather than a blank panel.

**Provenance.** Block `{bundle['seed']}`, the same block PermGraph, Frontier and DisplacementField
pin. The shipped neighbourhood is every block within {HOPS} block-adjacency hops of the seed --
{len(bundle['blocks']):,} blocks, of which {holed} carry an interior ring. {HOPS} hops is not a
round number with margin added: it is the smallest radius that contains the seed's own accretion at
the slider's maximum budget, so the widget can never grow a region with a hole in it where shipped
data runs out.

**The budget slider** runs {budget['min']:,} to {budget['max']:,} buildings in steps of
{budget['step']:,}, defaulting to {budget['default']:,}. The floor is deliberate, not arbitrary: at
{budget['min']:,} buildings the region is the seed alone -- shown, not hidden.

**Reference cases.** `hood.json` carries {len(bundle['reference'])} seed/budget pairs, each with
`DenseClusterRegionBuilder`'s own accretion order for it. They are what
`tests/test_region_grow_bundle.py` re-derives against live Python, pin the TypeScript greedy to
production rather than to a re-implementation of its rule, and confirm growth is NESTED: a bigger
budget's order always extends a smaller one's for the same seed.

| seed | max_buildings | blocks | buildings |
|---|---|---|---|
{rows}

Regenerate: `pixi run python -m scripts.gen_region_grow`
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)

    blocks = load_blocks(CITY)
    assert blocks.crs is not None and not blocks.crs.is_geographic, (
        "load_blocks must project -- dwithin(STREET_TOL) is ~55 km in lon/lat (design §1.5)")
    epsg = blocks.crs.to_epsg()
    if epsg is None:
        raise ValueError(f"{blocks.crs} has no EPSG code")

    all_ids = cast(list[str], list(blocks["block_id"]))
    idx_by_id = {b: i for i, b in enumerate(all_ids)}
    counts = [0.0 if pd.isna(c) else float(c) for c in blocks["building_count"]]
    geoms = list(blocks.geometry)
    log.info("loaded %d blocks above MIN_COUNT=%d", len(all_ids), MIN_COUNT)

    ids = neighbourhood(blocks, SEED, hops=HOPS)
    log.info("%d-hop neighbourhood of %s: %d blocks", HOPS, SEED, len(ids))

    reference = [growth(blocks, s, b) for s in REFERENCE_SEEDS for b in REFERENCE_BUDGETS]

    # The containment assertion the spec insists is ASSERTED, not reasoned about: a 54-block
    # accretion could in principle reach 53 hops, so block COUNTS alone (54 blocks vs. a 213-block
    # hood) prove nothing about whether every one of them actually lands IN the hood.
    full = next(c for c in reference if c["seed"] == SEED and c["max_buildings"] == BUDGET.max)
    hood_set = set(ids)
    missing = [b for b in full["order"] if b not in hood_set]
    if missing:
        raise ValueError(
            f"growth at the slider's maximum budget ({BUDGET.max}) leaves the shipped "
            f"{HOPS}-hop neighbourhood: {len(missing)} of {len(full['order'])} blocks are not in "
            f"it ({missing[:5]}...). The widget would draw a region with holes in it. Raise HOPS "
            f"until this passes -- do NOT lower BUDGET.max, which would hide the widget's most "
            f"interesting range. Block counts do not settle this: a 54-block accretion can in "
            f"principle reach 53 hops, which is why this is asserted and not inferred.")

    # Global (city-wide, RAW geometry) adjacency -- the SAME basis `DenseClusterRegionBuilder`
    # scores against internally, so `area_m2`/`perimeter_m` below reproduce its depth-proxy exactly
    # at every budget the slider can reach, not just the 12 pinned `reference` cases.
    adj_global = _block_adjacency(geoms)
    local_idx = {b: k for k, b in enumerate(ids)}
    ox = float(min(geoms[idx_by_id[b]].bounds[0] for b in ids))
    oy = float(min(geoms[idx_by_id[b]].bounds[1] for b in ids))

    hood_blocks: list[HoodBlock] = []
    simplified_geoms = []
    for b in ids:
        i = idx_by_id[b]
        raw = geoms[i]
        simp = raw.simplify(SIMPLIFY_M)
        simplified_geoms.append(simp)
        nbrs = sorted(local_idx[all_ids[j]] for j in adj_global[i] if all_ids[j] in hood_set)
        hood_blocks.append(HoodBlock(
            block_id=b, n=int(counts[i]), area_m2=sigfig(float(raw.area)),
            perimeter_m=sigfig(float(raw.length)),
            rings=polygon_rings(simp, ox, oy, what=f"block {b!r}"), adj=nbrs))
    log.info("holed blocks: %d of %d", sum(1 for hb in hood_blocks if len(hb["rings"]) > 1),
             len(hood_blocks))

    hood_gdf = gpd.GeoDataFrame({"block_id": ids}, geometry=simplified_geoms, crs=blocks.crs)

    boot = next(c for c in reference if c["seed"] == SEED and c["max_buildings"] == BUDGET.default)
    boot_ids = set(boot["order"])
    boot_local = {local_idx[b] for b in boot_ids}
    frontier_local = {j for i in boot_local for j in hood_blocks[i]["adj"]} - boot_local
    frontier_ids = {ids[j] for j in frontier_local}

    fig = _render_hood(hood_gdf, boot_ids, frontier_ids, SEED)
    save_render(fig, OUT / "hood.png")
    plt.close(fig)
    log.info("wrote %s", OUT / "hood.png")

    bundle = HoodBundle(
        city=CITY, seed=SEED, origin=[ox, oy], crs_epsg=epsg, blocks=hood_blocks,
        budget=BudgetDict(min=BUDGET.min, max=BUDGET.max, step=BUDGET.step,
                          default=BUDGET.default),
        encoding=EncodingDict(hood_color=ENCODING.hood_color, hood_lw=ENCODING.hood_lw,
                              region_color=ENCODING.region_color, region_lw=ENCODING.region_lw,
                              region_alpha=ENCODING.region_alpha,
                              seed_color=ENCODING.seed_color,
                              frontier_color=ENCODING.frontier_color, pad=ENCODING.pad),
        reference=reference,
    )
    (OUT / "hood.json").write_text(json.dumps(bundle) + "\n", encoding="utf-8")
    log.info("wrote %s (%.1f KB)", OUT / "hood.json", (OUT / "hood.json").stat().st_size / 1024.0)

    DTS.parent.mkdir(parents=True, exist_ok=True)
    DTS.write_text(DTS_TEMPLATE, encoding="utf-8")
    log.info("wrote %s", DTS)

    (OUT / "README.md").write_text(readme_markdown(bundle), encoding="utf-8")
    log.info("wrote %s", OUT / "README.md")


if __name__ == "__main__":
    main()
