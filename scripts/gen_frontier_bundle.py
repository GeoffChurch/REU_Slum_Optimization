"""Bake the per-method frontier table for the Methods index page's Frontier widget.

Every method's FULL drainage-ordered prefix table on the pinned block: road length, displacement and
permeability at m = 0..R. Both axes, because a target set on either one is answered against the
other -- and both are monotone in m, so the browser answers "which methods clear this target, and at
what least road" by the same binary search `budget.prefix_to_permeability` runs in Python, over the
same sequence. Not an interpolation.

784 prefixes across 8 methods, ~20 s of permeability solving. The parent design called full prefix
tables the long pole; at block scale they are not one. The ~12 hour figure belongs to regions.

A SEPARATE bundle from examples/perm-graph/bundle.json: same block, different page, and folding
these curves in would make the Methods index download 278 KB of per-prefix potentials and currents
to draw a chart.

Run:  pixi run python -m scripts.gen_frontier_bundle
"""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.colors import to_hex

from reblock.budget import building_radii, displacement, street_first_ordered
from reblock.compare import load_permeability_config
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency
from reblock.emit import (
    FRONTIER_GUIDE_COLOR,
    FRONTIER_GUIDE_LW,
    FRONTIER_LW,
    FRONTIER_X_LABEL,
    FRONTIER_Y_LABEL,
    method_colors,
)
from reblock.method_labels import friendly_method_name
from reblock.permeability import egress_power, permeability
from scripts._example_block import load_example_block

OUT = Path("examples/method-comparison/frontier.json")
DTS = Path("web/src/frontier.d.ts")
SIGFIGS = 6

# What the widget DRAWS with, baked here rather than chosen in the TypeScript or restated on the
# page. Five of these come straight from reblock.emit, which draws the fallback PNG the widget
# replaces, so the two charts agree by construction and not by two lists being kept in step: a
# reader with JS off and a reader with JS on must not see different charts. The rest are the web
# chart's own affordances, which the PNG has no equivalent for.
#
# PERCENT: emit.compare_report puts a PercentFormatter(xmax=1) on BOTH axes and writes its guide
# legend entries as `{:.0%}`, so the widget's ticks, targets and guide labels are percentages too --
# it previously drew the same numbers as bare fractions (0.6 where the PNG says 60%), which is
# exactly the JS-off/JS-on divergence this block exists to prevent.
CHART = {
    "x_label": FRONTIER_X_LABEL,
    "y_label": FRONTIER_Y_LABEL,
    "line_width": FRONTIER_LW,
    "guide_colour": FRONTIER_GUIDE_COLOR,
    "guide_width": FRONTIER_GUIDE_LW,
    # emit's guides are `ls="--"`; this is that dash, spelled as SVG's own dash-array.
    "guide_dash": "6 4",
    # niceTicks target. 5 puts the x ticks on 0/10/20/30/40% and the y ticks on 0/20/../100%, so
    # the extreme ticks land exactly on the axis ends -- which is what makes svg.ts's plot rect
    # (recovered FROM the tick extremes) the true data area rather than an inset of it.
    "tick_target": 5,
    # The gutter svg.ts's drawAxes draws tick labels and axis titles in, as a fraction of the box.
    # 0.15, not fitAxes's 0.04 default: on a 300 px box the default reserves under one em and the
    # labels land back on top of the plot. 0.15 is the value web/test/svg.test.ts pins as
    # GENEROUS_PAD, and the only nonzero pad whose label containment is under test.
    "pad": 0.15,
    # Slider step and drag quantisation: one percentage point of the axis. A whole percent, so every
    # target the widget can be set to prints exactly under emit's own `{:.0%}` -- a finer step would
    # make the guide label round to a percentage the guide is not actually at.
    "slider_step": 0.01,
    # Permeability is a fraction of parcels and the y axis is all of it. (The PNG's y axis
    # autoscales to the data instead -- ~0.99 here -- so this is the one axis end that is not
    # identical to the PNG's; it is a superset of it, never a crop.)
    "permeability_max": 1.0,
}


def _r(x: float) -> float:
    """6 significant digits. Safe here because every value is a metric in [0, 1] or a road length in
    the hundreds -- unlike coordinates, which need absolute rounding (see gen_web_bundle._c)."""
    return float(f"%.{SIGFIGS}g" % x)


def main() -> None:
    block, roads_by_method = load_example_block()
    pcfg = load_permeability_config()
    params = pcfg.params

    # Both frozen once and threaded through every solve: functions of block geometry alone.
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    p0, _ = egress_power(block, None, params, adj=adj)
    radii = building_radii(block.building_points)
    n_buildings = len(block.building_points)

    # Curve colours keyed exactly as the fallback PNG keys them: `method_colors` over the SAME
    # ordered method list this run draws, which is what run_permeability_lenses hands
    # emit.compare_report as `method_order` (list(methods) -- the selected set, in order). Hex,
    # because an SVG stroke needs a CSS colour where matplotlib takes an RGB triple.
    colours = {name: to_hex(rgb) for name, rgb in method_colors(list(roads_by_method)).items()}

    methods: dict[str, object] = {}
    for name, roads in roads_by_method.items():
        ordered = street_first_ordered(block, roads, STREET_TOL)
        road_m, disp, perm = [], [], []
        for m in range(len(ordered) + 1):
            prefix = ordered.iloc[:m]
            road_m.append(_r(float(prefix.geometry.length.sum())))
            disp.append(_r(displacement(block.building_points, radii, prefix) / n_buildings
                           if n_buildings else 0.0))
            perm.append(_r(permeability(block, prefix, params, p0=p0, adj=adj)))
        methods[name] = {
            "road_m": road_m, "displacement": disp, "permeability": perm,
            # The legend name and the curve colour travel WITH the curve: the widget iterates the
            # bundle's own keys, so a method added to this bake cannot reach the chart unlabelled or
            # uncoloured, and neither can be reconstructed from a raw key like
            # `greedy_arterial_access_displacement`.
            "label": friendly_method_name(name), "colour": colours[name],
        }

    bundle = {
        "block_id": block.block_id,
        # The widget boots with its target lines here, matching the dashed guides on the fallback
        # PNG -- whose caption the reader is looking at. Read from conf, never typed.
        "matched_displacement": pcfg.matched_displacement,
        "matched_permeability": pcfg.matched_permeability,
        "frontier_xmax": pcfg.frontier_xmax,
        "chart": CHART,
        "methods": methods,
    }
    OUT.write_text(json.dumps(bundle) + "\n", encoding="utf-8")
    DTS.write_text(DTS_TEMPLATE, encoding="utf-8")


DTS_TEMPLATE = '''// GENERATED by scripts/gen_frontier_bundle.py -- do not edit.
// Regenerate: pixi run python -m scripts.gen_frontier_bundle
export interface MethodCurve {
  road_m: number[];
  displacement: number[];
  permeability: number[];
  label: string;
  colour: string;
}
export interface ChartStyle {
  x_label: string;
  y_label: string;
  line_width: number;
  guide_colour: string;
  guide_width: number;
  guide_dash: string;
  tick_target: number;
  pad: number;
  slider_step: number;
  permeability_max: number;
}
export interface FrontierBundle {
  block_id: string;
  matched_displacement: number;
  matched_permeability: number;
  frontier_xmax: number;
  chart: ChartStyle;
  methods: Record<string, MethodCurve>;
}
'''


if __name__ == "__main__":
    main()
