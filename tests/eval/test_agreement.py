from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely import affinity
from shapely.geometry import LineString

from reblock.eval.agreement import buffered_iou, directional_chamfer

CRS_M = CRS.from_epsg(32734)


def _net(*geoms: LineString) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=list(geoms), crs=CRS_M)


def test_identical_networks_score_perfectly() -> None:
    net = _net(LineString([(0, 0), (100, 0)]), LineString([(50, -50), (50, 50)]))
    assert buffered_iou(net, net) == pytest.approx(1.0)
    precision, recall = directional_chamfer(net, net)
    assert precision == pytest.approx(0.0, abs=1e-6)
    assert recall == pytest.approx(0.0, abs=1e-6)


def test_disjoint_networks_score_zero_iou() -> None:
    a = _net(LineString([(0, 0), (100, 0)]))
    b = _net(LineString([(0, 10_000), (100, 10_000)]))
    assert buffered_iou(a, b) == pytest.approx(0.0)


def test_iou_decays_monotonically_with_offset() -> None:
    """A single far-offset case is vacuous -- with r=3 the buffers stop overlapping past 6 m and
    score 0 for ANY metric. The graded series is what actually tests the metric.

    Deliberately NOT parametrized: the property under test is the ORDERING across offsets, which
    a per-offset parametrization cannot express (each case would only see its own score)."""
    ref = _net(LineString([(0, 0), (100, 0)]))
    scores = {
        o: buffered_iou(
            _net(cast(LineString, affinity.translate(ref.geometry.iloc[0], yoff=o))), ref, r=3.0
        )
        for o in (0.0, 1.0, 3.0, 6.0, 12.0)
    }
    assert scores[0.0] > scores[1.0] > scores[3.0] > scores[6.0]
    assert scores[0.0] == pytest.approx(1.0)
    assert scores[6.0] == pytest.approx(0.0)
    assert scores[12.0] == pytest.approx(0.0)


def test_iou_at_offset_equal_to_radius_is_a_pinned_value() -> None:
    ref = _net(LineString([(0, 0), (100, 0)]))
    prop = _net(cast(LineString, affinity.translate(ref.geometry.iloc[0], yoff=3.0)))
    assert buffered_iou(prop, ref, r=3.0) == pytest.approx(0.33, abs=0.03)


def test_chamfer_is_asymmetric_for_a_strict_subset() -> None:
    """The only thing directional Chamfer exists to expose. A proposal covering half the
    reference has near-zero precision error but large recall error; the transpose flips it."""
    reference = _net(LineString([(0, 0), (100, 0)]))
    proposal = _net(LineString([(0, 0), (50, 0)]))

    precision, recall = directional_chamfer(proposal, reference)
    assert precision == pytest.approx(0.0, abs=0.5)
    assert recall > 5.0

    precision_t, recall_t = directional_chamfer(reference, proposal)
    assert recall_t == pytest.approx(0.0, abs=0.5)
    assert precision_t > 5.0


def test_chamfer_densification_step_is_a_quantization_floor() -> None:
    """A 2 m step puts a ~1 m floor on Chamfer; a finer step must not raise the score."""
    ref = _net(LineString([(0, 0), (100, 0)]))
    prop = _net(cast(LineString, affinity.translate(ref.geometry.iloc[0], yoff=4.0)))
    coarse, _ = directional_chamfer(prop, ref, step=2.0)
    fine, _ = directional_chamfer(prop, ref, step=0.5)
    assert fine <= coarse + 1e-6
    assert fine == pytest.approx(4.0, abs=0.2)


def test_empty_proposal_scores_zero_iou() -> None:
    assert buffered_iou(_net(), _net(LineString([(0, 0), (100, 0)]))) == pytest.approx(0.0)
