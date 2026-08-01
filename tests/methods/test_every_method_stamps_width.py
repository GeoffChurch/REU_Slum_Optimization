"""Every configured method must emit roads the metric can actually score.

`euclidean_grid` shipped for a day without a `width_m` column: the width refactor stamped eight
methods by hand and missed it, and no existing test noticed because nothing scored that method's
output. It is in the flagship examples lineup, so the examples pipeline would have crashed on the
next regeneration.

The lesson is that per-method tests cannot cover a per-method OBLIGATION -- a new method, or one the
next refactor skips, is exactly what slips through. This walks `conf/compare_config.yaml`'s own
`all_methods` so the guard's coverage grows with the config rather than with anyone's diligence.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.budget import building_radii, displacement
from reblock.contracts import Block
from reblock.permeability import ONEWAY_COL, WIDTH_COL, PermeabilityParams, permeability

UTM = CRS.from_epsg(32734)
PARAMS = PermeabilityParams()

# Needs a live OSM/Overpass fetch, so it cannot run offline; its width is asserted by
# tests/methods/test_osm_footpaths.py instead.
NEEDS_NETWORK = {"osm_footpaths", "demand_greedy"}


def _block(k: int = 8, cell: float = 10.0) -> Block:
    polys, ids = [], []
    for r in range(k):
        for c in range(k):
            x0, y0 = c * cell, r * cell
            polys.append(Polygon([(x0, y0), (x0 + cell, y0), (x0 + cell, y0 + cell),
                                  (x0, y0 + cell)]))
            ids.append(r * k + c)
    return Block(
        block_id="stamp", crs=UTM,
        boundary=Polygon([(0, 0), (k * cell, 0), (k * cell, k * cell), (0, k * cell)]),
        parcels=gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM),
        streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (k * cell, 0)])], crs=UTM),
        building_points=gpd.GeoDataFrame(geometry=[p.centroid for p in polys], crs=UTM))


def _configured_methods() -> list[str]:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=["shapefile=x"])
    names = sorted(OmegaConf.to_container(cfg.all_methods, resolve=False))  # type: ignore[arg-type]
    return [n for n in names if n not in NEEDS_NETWORK]


@pytest.mark.parametrize("name", _configured_methods())
def test_method_emits_scorable_roads(name: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=["shapefile=x"])
    method = instantiate(cfg.all_methods[name])

    block = _block()
    roads = method.propose(block).roads
    if roads is None or len(roads) == 0:
        pytest.skip(f"{name} proposed nothing on the synthetic block")

    assert WIDTH_COL in roads.columns, f"{name} emits roads without a '{WIDTH_COL}' column"
    widths = roads[WIDTH_COL].to_numpy(dtype=float)
    oneway = (roads[ONEWAY_COL].to_numpy(dtype=bool) if ONEWAY_COL in roads.columns
              else np.zeros(len(roads), dtype=bool))
    floors = np.where(oneway, PARAMS.min_one_way_width_m, PARAMS.min_two_way_width_m)
    assert (widths >= floors).all(), f"{name} emits a road too narrow for its own direction"

    # ...and both scorers actually accept it -- the assertion that fails on a missing stamp, and
    # why a column check alone would not be enough (the stamp has to survive the ORDERING the lenses
    # apply, not merely exist on the emitted frame).
    #
    # The bound is -1e-9, not 0: permeability is `1 - P(roads)/P(no roads)`, so a method whose roads
    # happen to buy nothing on this uniform synthetic grid lands on zero plus sparse-solve rounding
    # (flow_paths reads -1.1e-13). That is the metric's floating-point zero, not a monotonicity
    # violation, and this test is not the place to audit any method's efficacy on a toy fixture.
    assert -1e-9 <= float(permeability(block, roads, PARAMS)) <= 1.0
    assert displacement(block.building_points, building_radii(block.building_points), roads) >= 0.0
