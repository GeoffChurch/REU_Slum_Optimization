import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import box

from reblock.data.settlements import exclusion_holdout, settlement_labels

CRS_M = CRS.from_epsg(32734)
CRS_GEO = CRS.from_epsg(4326)


def _blocks(*offsets: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[box(x, 0, x + 50, 50) for x in offsets], crs=CRS_M)


def test_settlement_labels_group_near_blocks_and_split_far_ones() -> None:
    blocks = _blocks(0, 60, 5000)          # first two within 100 m, third far away
    labels = settlement_labels(blocks, tol_m=100.0)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_settlement_labels_chain_transitively() -> None:
    """Documents the pathology that demotes this to a reporting label: A-B and B-C within
    tolerance puts A and C in one settlement however far apart they are."""
    blocks = _blocks(0, 60, 120, 180)
    assert len(set(settlement_labels(blocks, tol_m=100.0))) == 1


def test_exclusion_holdout_drops_everything_inside_the_radius() -> None:
    blocks = _blocks(0, 60, 5000)
    donors = exclusion_holdout(blocks, recipient_idx=0, radius_m=100.0)
    assert donors == [2]


def test_exclusion_holdout_never_returns_the_recipient() -> None:
    blocks = _blocks(0, 5000)
    assert 0 not in exclusion_holdout(blocks, recipient_idx=0, radius_m=0.0)


def test_exclusion_holdout_is_monotone_in_radius() -> None:
    """The property that makes this a defensible fold definition: more radius, never more donors."""
    blocks = _blocks(0, 60, 200, 5000)
    counts = [len(exclusion_holdout(blocks, 0, radius_m=r)) for r in (0.0, 100.0, 500.0, 10_000.0)]
    assert counts == sorted(counts, reverse=True)


def test_exclusion_holdout_raises_on_geographic_crs() -> None:
    """A geographic CRS makes `radius_m=2000` compare against a degree-valued distance (always
    << 2000), so every block reads as eligible -- maximum donor leakage presented as a holdout.
    Must raise rather than silently returning that."""
    blocks = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)], crs=CRS_GEO)
    with pytest.raises(ValueError, match="projected"):
        exclusion_holdout(blocks, recipient_idx=0, radius_m=2000.0)


def test_exclusion_holdout_raises_on_missing_crs() -> None:
    blocks = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)])
    with pytest.raises(ValueError, match="projected"):
        exclusion_holdout(blocks, recipient_idx=0, radius_m=2000.0)


def test_settlement_labels_raises_on_geographic_crs() -> None:
    """A geographic CRS makes `tol_m=100` compare against a degree-valued `dwithin`, which never
    binds at realistic block spacing -- collapsing everything into one settlement."""
    blocks = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)], crs=CRS_GEO)
    with pytest.raises(ValueError, match="projected"):
        settlement_labels(blocks, tol_m=100.0)


def test_settlement_labels_raises_on_missing_crs() -> None:
    blocks = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)])
    with pytest.raises(ValueError, match="projected"):
        settlement_labels(blocks, tol_m=100.0)
