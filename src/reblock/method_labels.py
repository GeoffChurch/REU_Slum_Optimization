"""Human-facing display names for method keys. Pure display layer: internal method keys are used
for config, filenames, and caching and must never be renamed -- this module only maps a key to a
friendlier label for readers (frontier-plot legend, generated READMEs). STDLIB-ONLY so it stays
cheap to import from stdlib-only callers (e.g. `scripts/gen_example_readme.py`) -- do not add
geopandas/matplotlib/pandas imports here.
"""
from __future__ import annotations

FRIENDLY_METHOD_NAMES = {
    "osm_footpaths": "OSM Footpaths",
    "clearance_looped": "Looped Tree",
    "euclidean_grid": "Grid",
    "greedy_arterial_repulsion": "Throughways",
    "demand_greedy": "Desire-Line Tree",
    "demand_greedy_uniform": "Plain Tree",
    "demand_looped": "Looped Plain Tree",
    "clearance_looped_cheap": "Looped Tree (cheap loops)",
    "flow_paths": "Worn Paths",
    "flow_paths_noreinforce": "Worn Paths (no feedback)",
}


def friendly_method_name(key: str) -> str:
    """Human-facing method label; falls back to the raw key for unmapped methods."""
    return FRIENDLY_METHOD_NAMES.get(key, key)
