"""Bake the browser bundle for the Permeability page's PermGraph widget.

Everything the widget draws is computed HERE, in Python, and read there: geometry, the
per-prefix fields, the colour ramp, and the width rules. The widget's only freedoms are which
prefix and which layer. That is deliberate -- the widget replaces a PNG whose caption quotes
numbers from `perm_graph.json`, so a second opinion about how to draw the same data would put two
pictures under one caption.

Committed, not built in CI: `scripts/gen_site_pages.py` is stdlib-only (CI builds the site with
only mkdocs-material installed) and baking needs geopandas and the solver.

Run:  pixi run python -m scripts.gen_web_bundle
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
from geopandas import GeoDataFrame
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from matplotlib import colormaps
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry

from reblock.budget import prefix_to_permeability, street_first_ordered
from reblock.compare import load_permeability_config
from reblock.contracts import Block, Method, Screen, Source
from reblock.derivations import propose
from reblock.derive.access import STREET_TOL
from reblock.perm_graph import permeability_graph
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder
from reblock.render import (
    _BOUNDARY_COLOR,
    _CONTEXT_OUTLINE,
    _EDGE_GREY,
    _EDGE_LW_MAX,
    _EDGE_LW_MIN,
    _NODE_RADIUS_FRAC,
    _PERM_CMAP,
    _ROAD_COLOR,
    _UPGRADED_LW,
)

log = logging.getLogger(__name__)

VARIANT = "method_comparison"      # pins ZAF.9.3.1_1_40972; see conf/example/method_comparison.yaml
METHOD = "clearance"
OUT = Path("examples/perm-graph")
DTS = Path("web/src/bundle.d.ts")
SIGFIGS = 6


def load_block_and_roads() -> tuple[Block, GeoDataFrame]:
    """The pinned block and `clearance`'s full road set. A FUNCTION rather than inline setup because
    tests/test_web_bundle.py's parity test re-derives the same inputs to check the committed bundle
    against them -- if the test loaded the block a second, independent way, the two could drift and
    the parity check would be comparing the wrong things."""
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config",
                      overrides=[f"+example={VARIANT}", "data=capetown_full"])
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    groups = [list(g) for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, int(cfg.max_blocks))[0]
    assert len(region) == 1, "this figure set is single-block by design"
    block = region[0]

    method = cast(Method, instantiate(cfg.all_methods[METHOD]))
    roads = cast(GeoDataFrame, propose(method, block).roads)
    return block, roads


def _r(x: float) -> float:
    """FIELD VALUES at 6 significant digits -- far beyond what a canvas shows or the readout
    quotes, and it keeps the payload near 300 KB. tests/test_web_bundle.py's parity assertion is
    stated at this precision, so changing it means changing that tolerance too.

    NOT for coordinates -- see `_c`."""
    return float(f"%.{SIGFIGS}g" % x)


def _c(x: float) -> float:
    """COORDINATES, as centimetres of absolute precision.

    Significant digits are the wrong tool here and dangerously so: a Cape Town UTM northing is
    ~6,240,000, so `%.6g` would round it to the nearest 10 METRES and dissolve the parcel geometry.
    Coordinates are emitted relative to `origin` (see below), which both fixes the precision problem
    and shrinks the payload, since local metres are 3-4 digits instead of 7."""
    return round(x, 2)


def _line_coords(geom: BaseGeometry, ox: float, oy: float) -> list[list[list[float]]]:
    """Explode a LineString/MultiLineString into one coordinate list per component, at the same
    centimetre precision `_c` gives every other coordinate in this bundle (see its docstring): a
    street's northing is exactly as far from the origin as a parcel's, and significant-digit
    rounding would dissolve it the same way. `Block.streets` is documented as line geometry
    (`_draw_boundary_and_streets` in render.py draws it with no other case); anything else is a
    contract violation worth raising on, not silently dropping."""
    if isinstance(geom, LineString):
        lines: list[LineString] = [geom]
    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
    else:
        raise ValueError(
            f"unexpected street geometry type {geom.geom_type!r} -- report this instead of "
            f"silently dropping it")
    return [[[_c(x - ox), _c(y - oy)] for x, y in line.coords] for line in lines]


def _ramp(name: str, n: int = 256) -> list[str]:
    """The colormap sampled to hex stops. `_PERM_CMAP` is the STRING "YlOrRd" -- a matplotlib
    colormap name -- and the browser has no matplotlib, so a hand-rolled JS approximation would put
    the same block in two palettes on one page (the drift the parent design warns about)."""
    cmap = colormaps[name]
    stops = []
    for t in np.linspace(0.0, 1.0, n):
        r, g, b = (int(round(c * 255)) for c in cmap(t)[:3])
        stops.append(f"#{r:02x}{g:02x}{b:02x}")
    return stops


DTS_TEMPLATE = '''// GENERATED by scripts/gen_web_bundle.py -- do not edit.
// Regenerate: pixi run python -m scripts.gen_web_bundle
// This file is what makes a renamed Python field a TypeScript error instead of a blank panel.
export interface Encoding {
  width_norm: { conductance: number; current: number };
  edge_lw_min: number;
  edge_lw_max: number;
  upgraded_lw: number;
  node_radius_frac: number;
  ramp: string[];
  road_color: string;
  boundary_color: string;
  parcel_color: string;
  edge_color: string;
}
export interface Bundle {
  block_id: string;
  method: string;
  lens_b_index: number;
  n_prefixes: number;
  /** UTM easting/northing subtracted from every coordinate below; all geometry is local metres. */
  origin: [number, number];
  parcels: [number, number][][];
  /** Block exterior ring, relative to `origin` -- fallback-parity background layer; see
   * `_draw_boundary_and_streets` in render.py, which draws this under the graph on every PNG. */
  boundary: [number, number][];
  /** Existing street network, relative to `origin`; one entry per disjoint line (a block's
   * streets are not always a single connected LineString). Fallback-parity, same as `boundary`. */
  streets: [number, number][][];
  nodes: { cx: number[]; cy: number[]; ground_g: number[] };
  edges: { rows: number[]; cols: number[]; footpath_g: number[]; first_upgraded_at: number[] };
  roads: { coords: [number, number][]; width_m: number }[];
  prefix: {
    potential: number[][];
    current: number[][];
    permeability: number[];
    road_m: number[];
  };
  encoding: Encoding;
}
'''


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)
    pcfg = load_permeability_config()
    params = pcfg.params

    block, roads = load_block_and_roads()

    ordered = street_first_ordered(block, roads, STREET_TOL)
    prefix, reached = prefix_to_permeability(block, roads, pcfg.matched_permeability, params,
                                            tol=STREET_TOL)
    if not reached:
        raise SystemExit(f"{METHOD} never reached P*={pcfg.matched_permeability}")
    # prefix_to_permeability returns `ordered.iloc[:lo]`, so its LENGTH is the index into the
    # canonical sequence -- this is the prefix graph_current_after.png shows, and the widget must
    # boot here or the caption below it describes a different picture.
    lens_b_index = len(prefix)

    figs = [permeability_graph(block, cast(GeoDataFrame, ordered.iloc[:m]), params)
            for m in range(len(ordered) + 1)]
    base = figs[0]

    # `upgraded` is monotone in the road set (conductance enters only through max(footpath, road)),
    # so store the first m at which each edge is raised instead of 21 x 745 booleans. -1 = never.
    first_upgraded_at = np.full(len(base.rows), -1, dtype=int)
    for m, f in enumerate(figs):
        newly = f.upgraded & (first_upgraded_at < 0)
        first_upgraded_at[newly] = m

    # Mesh-only width norms, matching render_graph's rule exactly (see gen_perm_graph.py): the
    # road-dominated max would collapse the mesh into a sub-pixel band.
    #
    # This deliberately differs from gen_perm_graph.py's pooling: piece B pools p99 over only two
    # prefixes (0 and the Lens-B index), because it only ever renders those two. This widget spans
    # every prefix 0..len(ordered), including ones piece B never rendered; a Lens-B-derived constant
    # would let mesh currents at prefixes past Lens B clip to maximum width -- the uniform-thick-
    # mesh legibility failure piece B spent three fix rounds removing. So the norm here pools over
    # ALL prefixes' mesh-only edges, using the full-network mesh mask (edges never raised by any
    # prefix, i.e. `~figs[-1].upgraded`) rather than each prefix's own (monotonically shrinking)
    # mask -- one fixed set of denominator edges, so the norm does not itself depend on the slider
    # position.
    mesh = ~figs[-1].upgraded
    width_norm = {
        "conductance": _r(float(np.percentile(np.abs(base.footpath_g[mesh]), 99))),
        "current": _r(float(np.percentile(
            np.abs(np.concatenate([f.current[mesh] for f in figs])), 99))),
    }

    # Everything geometric is emitted RELATIVE to this, in metres. The canvas works in local metres
    # and never learns the CRS; `width_m` is a length, so translation leaves it alone.
    ox, oy = float(base.cx.min()), float(base.cy.min())

    parcel_coords = []
    for g in block.parcels.geometry:
        # isinstance, not geom_type, so this line IS the runtime guard mypy can also verify: it
        # narrows `g` to Polygon, which is what makes `.interiors`/`.exterior` below type-check
        # instead of resolving through BaseGeometry, the union GeoSeries iteration yields.
        if not isinstance(g, Polygon):
            raise ValueError(
                f"block {block.block_id!r} has a non-Polygon parcel ({g.geom_type}) -- the "
                f"bundle format assumes every parcel is a simple Polygon with an exterior ring "
                f"and no holes; report this instead of silently dropping geometry")
        if len(g.interiors) != 0:
            raise ValueError(
                f"block {block.block_id!r} has a non-simple parcel (Polygon, "
                f"{len(g.interiors)} interior rings) -- the bundle format assumes every parcel "
                f"is a simple Polygon with an exterior ring and no holes; report this instead "
                f"of silently dropping geometry")
        parcel_coords.append([[_c(x - ox), _c(y - oy)] for x, y in g.exterior.coords])

    # Fallback parity: _draw_boundary_and_streets (render.py) draws the block outline and the
    # EXISTING street network under every graph PNG, including graph_current_after.png -- the
    # exact image this widget replaces. The widget draws neither unless the bundle carries the
    # geometry, so bake it here rather than let the interactive version silently omit context the
    # static fallback always shows.
    if not isinstance(block.boundary, Polygon):
        raise ValueError(
            f"block {block.block_id!r} has a non-Polygon boundary ({block.boundary.geom_type}) "
            f"-- load_block_and_roads asserts this figure set is single-block, so the gappy-"
            f"region MultiPolygon case _draw_boundary_and_streets skips should not arise here; "
            f"report this instead of silently dropping the boundary")
    boundary_coords = [[_c(x - ox), _c(y - oy)] for x, y in block.boundary.exterior.coords]

    street_coords: list[list[list[float]]] = []
    for g in block.streets.geometry:
        street_coords.extend(_line_coords(g, ox, oy))

    bundle = {
        "block_id": block.block_id,
        "method": METHOD,
        "lens_b_index": lens_b_index,
        "n_prefixes": len(figs),
        "origin": [ox, oy],
        "parcels": parcel_coords,
        "boundary": boundary_coords,
        "streets": street_coords,
        "nodes": {"cx": [_c(v - ox) for v in base.cx], "cy": [_c(v - oy) for v in base.cy],
                  "ground_g": [_r(v) for v in base.ground_g]},
        "edges": {"rows": base.rows.tolist(), "cols": base.cols.tolist(),
                  "footpath_g": [_r(v) for v in base.footpath_g],
                  "first_upgraded_at": first_upgraded_at.tolist()},
        "roads": [{"coords": [[_c(x - ox), _c(y - oy)] for x, y in g.coords],
                   "width_m": float(w)}
                  for g, w in zip(ordered.geometry, ordered["width_m"], strict=True)],
        "prefix": {
            "potential": [[_r(v) for v in f.potential] for f in figs],
            "current": [[_r(v) for v in f.current] for f in figs],
            "permeability": [_r(1.0 - f.p / base.p) for f in figs],
            # `ordered.geometry.iloc[:m]` types (wrongly) as a scalar BaseGeometry -- the same
            # geopandas-stub slice-collapse the `cast`s elsewhere on this page work around --
            # which then makes `.length` resolve to a single float instead of a Series. Slicing
            # the frame (not the geometry column) before reading `.length` sidesteps it and is
            # the same value: selecting first-m-then-geometry equals geometry-then-first-m.
            "road_m": [_r(float(cast(GeoDataFrame, ordered.iloc[:m]).length.sum()))
                       for m in range(len(figs))],
        },
        "encoding": {
            "width_norm": width_norm,
            "edge_lw_min": _EDGE_LW_MIN, "edge_lw_max": _EDGE_LW_MAX,
            "upgraded_lw": _UPGRADED_LW, "node_radius_frac": _NODE_RADIUS_FRAC,
            "ramp": _ramp(_PERM_CMAP),
            "road_color": _ROAD_COLOR, "boundary_color": _BOUNDARY_COLOR,
            "parcel_color": _CONTEXT_OUTLINE, "edge_color": _EDGE_GREY,
        },
    }
    (OUT / "bundle.json").write_text(json.dumps(bundle) + "\n", encoding="utf-8")
    log.info("wrote %s", OUT / "bundle.json")

    DTS.parent.mkdir(parents=True, exist_ok=True)
    DTS.write_text(DTS_TEMPLATE, encoding="utf-8")
    log.info("wrote %s", DTS)


if __name__ == "__main__":
    main()
