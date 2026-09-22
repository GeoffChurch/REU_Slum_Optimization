"""The tier ladder's rejections, asserted so CI watches them -- not merely documented.

Each `# type: ignore[arg-type]` is an ASSERTION that the call is a type error. `mypy --strict`
turns on `--warn-unused-ignores`, so if the ladder ever stopped rejecting one of these, the ignore
would become unused and the typecheck gate would FAIL. A guard nobody has watched fail guards
nothing; this one fails by construction the moment the protocol stops discriminating.

Confirmed broken-then-fixed during the migration: without the ignores mypy reports exactly these
three `arg-type` errors and none on `accepts_what_it_should`.
"""
from __future__ import annotations

import geopandas as gpd

from reblock.buildings import AreaDiscs, Extents, Footprints, Positions, Shapes, SpacingDiscs


def _needs_extent(b: Extents) -> int:
    return len(b.radii)


def _needs_shape(b: Shapes) -> int:
    return len(b.polygons)


def accepts_what_it_should(pts: gpd.GeoDataFrame, polys: gpd.GeoDataFrame) -> None:
    """Every tier that HAS the capability must pass -- otherwise the ladder rejects everything
    and the failures below would be vacuous."""
    _needs_extent(SpacingDiscs(pts))
    _needs_extent(AreaDiscs(pts))
    _needs_extent(Footprints(polys))
    _needs_shape(Footprints(polys))


def rejects_what_it_should(pts: gpd.GeoDataFrame, bare: Positions) -> None:
    """Each line below MUST be a type error; the ignore is the assertion."""
    _needs_extent(bare)                 # type: ignore[arg-type]  # positions have no size
    _needs_shape(SpacingDiscs(pts))     # type: ignore[arg-type]  # a disc has no outline
    _needs_shape(AreaDiscs(pts))        # type: ignore[arg-type]  # nor an equal-area disc
