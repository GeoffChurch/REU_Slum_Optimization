"""Human-facing display names for method keys. Pure display layer: internal method keys are used
for config, filenames, and caching and must never be renamed -- this module only maps a key to a
friendlier label for readers (frontier-plot legend, generated READMEs). STDLIB-ONLY so it stays
cheap to import from stdlib-only callers (e.g. `scripts/gen_example_readme.py`) -- do not add
geopandas/matplotlib/pandas imports here.
"""
from __future__ import annotations

FRIENDLY_METHOD_NAMES = {
    "osm_footpaths": "OSM Footpaths",
    "topology": "Topology",
    "clearance": "Least-Cost Tree",
    "clearance_looped": "Looped Tree",
    "euclidean_grid": "Grid",
    "greedy_arterial_repulsion": "Throughways",
    "greedy_arterial_access_repulsion": "Frontage (lane-priced)",
    "greedy_arterial_access_displacement": "Frontage (street-priced)",
    "resistance_greedy": "Direct Objective",
    "resistance_lp": "Direct Objective (LP)",
    "cycle_native": "Loop Network",
    "flow_paths": "Worn Paths",
    "flow_paths_noreinforce": "Worn Paths (no feedback)",
}


def friendly_method_name(key: str) -> str:
    """Human-facing method label; falls back to the raw key for unmapped methods."""
    return FRIENDLY_METHOD_NAMES.get(key, key)
