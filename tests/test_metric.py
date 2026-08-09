from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml
from pyproj import CRS
from shapely.geometry import Polygon, box

from reblock.metric import (
    DENSITY_COMPACTNESS_FLOOR,
    DEPTH_DENSITY_PROXY_FLOOR,
    Compactness,
    Count,
    Density,
    Depth,
    DepthProxy,
    Gate,
    Power,
    Product,
)

_UTM = CRS.from_epsg(32643)


def _blocks() -> gpd.GeoDataFrame:
    # two unit-ish squares with known n, A, P
    a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])       # A=4, P=8
    b = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])       # A=1, P=4
    return gpd.GeoDataFrame({"block_id": ["a", "b"], "building_count": [16.0, 4.0],
                             "block_area_m2": [4.0, 1.0]}, geometry=[a, b], crs=_UTM)


def test_primitive_proxy_and_fine_closed_forms() -> None:
    b = _blocks()
    # Depth proxy = sqrt(n*A)/P ; fine = the passed peel depth (columns ignored for depth factor)
    assert np.allclose(Depth().proxy(b).to_numpy(), [np.sqrt(16 * 4) / 8, np.sqrt(4 * 1) / 4])
    assert Depth().fine(7.0, 16.0, 4.0, 8.0) == 7.0
    assert Depth().needs_peel is True
    # Density = n/A (proxy == fine, closed form, no peel)
    assert np.allclose(Density().proxy(b).to_numpy(), [16 / 4, 4 / 1])
    assert Density().fine(0.0, 16.0, 4.0, 8.0) == 16 / 4
    assert Density().needs_peel is False
    # Compactness = A/P^2
    assert np.allclose(Compactness().proxy(b).to_numpy(), [4 / 64, 1 / 16])
    assert Compactness().fine(0.0, 16.0, 4.0, 8.0) == 4 / 64
    assert Compactness().needs_peel is False
    # Count = n (proxy == fine, no peel) -- a total-building-count factor for composition
    assert np.allclose(Count().proxy(b).to_numpy(), [16.0, 4.0])
    assert Count().fine(0.0, 16.0, 4.0, 8.0) == 16.0
    assert Count().needs_peel is False


def test_combinators_fold_proxy_fine_and_needs_peel() -> None:
    b = _blocks()
    dd = Product([Depth(), Density()])
    assert np.allclose(dd.proxy(b).to_numpy(),
                       Depth().proxy(b).to_numpy() * Density().proxy(b).to_numpy())
    assert dd.fine(7.0, 16.0, 4.0, 8.0) == 7.0 * (16 / 4)
    assert dd.needs_peel is True                        # OR over children
    dc = Product([Density(), Compactness()])
    assert dc.needs_peel is False                       # no Depth in the tree
    assert dc.fine(0.0, 16.0, 4.0, 8.0) == (16 / 4) * (4 / 64)
    # Power over a SUB-EXPRESSION: sqrt(depth*density)
    root = Power(Product([Depth(), Density()]), 0.5)
    assert root.fine(9.0, 16.0, 4.0, 8.0) == (9.0 * (16 / 4)) ** 0.5
    assert np.allclose(root.proxy(b).to_numpy(), dd.proxy(b).to_numpy() ** 0.5)
    # Count weights density x compactness by total building count -> n^2 / P^2
    ndc = Product([Count(), Density(), Compactness()])
    assert ndc.fine(0.0, 16.0, 4.0, 8.0) == 16.0 * (16 / 4) * (4 / 64)
    assert ndc.needs_peel is False


def test_identity_distinguishes_expressions() -> None:
    assert Product([Depth(), Density()]).identity != Product([Density(), Compactness()]).identity
    assert Depth().identity == Depth().identity
    assert Power(Depth(), 2.0).identity != Power(Depth(), 3.0).identity


def test_gate_absolute_and_percentile() -> None:
    scores = {"a": 10.0, "b": 5.0, "c": 1.0, "d": 0.5}
    assert Gate("absolute", 5.0).keep(scores) == {"a", "b"}         # >= 5
    assert Gate("percentile", 50.0).keep(scores) == {"a", "b"}      # top 50%
    assert Gate("percentile", 25.0).keep(scores) == {"a"}           # top 25%


def test_config_floor_matches_python_definition() -> None:
    """conf/metric/density_compactness.yaml's absolute gate and DENSITY_COMPACTNESS_FLOOR are one
    number in two files. Same mirror-plus-drift-guard as the footpath tag list -- a screen whose
    config and code disagree about the floor silently selects a different population than the one
    every calibration number was measured on.

    The gate must also stay ABSOLUTE: a percentile re-defines the population with the corpus
    (Cape Town's old percentile-30 cut selects 7.6% of the ZAF+KEN corpus), which is exactly what
    calibrating an absolute floor was meant to end.
    """
    conf = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "conf/metric/density_compactness.yaml").read_text())
    gate = conf["metric_gate"]
    assert gate["kind"] == "absolute"
    assert float(gate["value"]) == DENSITY_COMPACTNESS_FLOOR


def test_depth_density_proxy_floor_matches_python_definition() -> None:
    """Same mirror-plus-drift-guard as the density_compactness floor, for the metric that replaced
    it as the default. Also pins the two properties the default rests on:

    * the gate is ABSOLUTE, so the population does not move with the corpus;
    * the metric does NOT need a peel -- that is the entire point of `DepthProxy` over `Depth`, and
      a config that silently reintroduced `Depth` would turn every screen run into a fine pass.

    FAULT INJECTION: changing the yaml value to 0.02 fails the value assertion; swapping
    `DepthProxy` for `Depth` in the yaml fails the needs_peel assertion.
    """
    conf = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "conf/metric/depth_density_proxy.yaml").read_text())
    gate = conf["metric_gate"]
    assert gate["kind"] == "absolute"
    assert float(gate["value"]) == DEPTH_DENSITY_PROXY_FLOOR

    targets = [t["_target_"] for t in conf["metric"]["terms"]]
    assert targets == ["reblock.metric.DepthProxy", "reblock.metric.Density"]
    metric = Product(name="depth_density_proxy", terms=(DepthProxy(), Density()))
    assert not metric.needs_peel, "the default screen metric must never trigger a fine pass"


def test_depth_proxy_fine_equals_its_own_proxy() -> None:
    """`DepthProxy` exists so the cheap estimator can be used with no peel, which only works if its
    scalar `fine` agrees with its vectorized `proxy`. `Depth`'s two differ BY DESIGN (proxy
    estimates, fine peels); this one's must not.

    FAULT INJECTION: returning `count / perim` from `fine` breaks the equality.
    """
    blocks = gpd.GeoDataFrame(
        {"building_count": [40.0, 120.0], "block_area_m2": [4000.0, 9000.0]},
        geometry=[box(0, 0, 40, 100), box(0, 0, 90, 100)], crs="EPSG:32734")
    m = DepthProxy()
    vec = m.proxy(blocks).to_numpy()
    for i, (count, area, perim) in enumerate(
            ((40.0, 4000.0, blocks.geometry.iloc[0].length),
             (120.0, 9000.0, blocks.geometry.iloc[1].length))):
        assert m.fine(0.0, count, area, perim) == pytest.approx(vec[i])
