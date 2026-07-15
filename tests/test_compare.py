import csv
import subprocess
import sys
from pathlib import Path


def test_compare_writes_frontier_and_curves(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare",
         "data=dji", "eval=kcomplexity", "max_blocks=1",
         "methods=[clearance,greedy_arterial_buildable]", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "frontier_access.csv").read_text()
    assert "clearance" in table and "greedy_arterial_buildable" in table
    assert list(tmp_path.glob("curve_access_*.png"))


def test_compare_displacement_cost_axis_runs_and_writes_curves(tmp_path: Path) -> None:
    # cost=displacement grades methods on the buildings-displaced x-axis (sparse methods land near
    # 0). clearance (fast) proves the axis is reachable end-to-end; the axis arithmetic is
    # unit-tested in test_budget, and greedy_arterial_displacement is the slow flagship.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "max_blocks=1", "methods=[clearance]", "cost=displacement", "corridor_m=3.0",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    # displacement uses a tradeoff table (terminal benefit + buildings displaced), NOT the length
    # frontier -- AUC inverts on the displacement axis (a home-sparing method scores 0).
    assert (tmp_path / "tradeoff_table_directness.csv").exists()
    assert not (tmp_path / "frontier_directness.csv").exists()
    assert "buildings_displaced" in (tmp_path / "tradeoff_table_directness.csv").read_text()
    assert list(tmp_path.glob("curve_directness_*.png"))


def test_compare_emits_per_metric_frontier_tables(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "max_blocks=1", "methods=[clearance,greedy_arterial_buildable]",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    for metric in ("access", "efficiency", "directness"):
        assert (tmp_path / f"frontier_{metric}.csv").exists()
        assert list(tmp_path.glob(f"curve_{metric}_*.png"))


def test_compare_singleton_via_explicit_block_ids_matches_plain_single_block(
    tmp_path: Path,
) -> None:
    # An explicit list-of-lists with ONE singleton group takes build_regions's
    # region_builder-expansion branch (not the classic screen=identity/None fallback) -- but a
    # singleton region is still the EXACT pre-region single-block path, keyed by the plain block_id.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare",
         "data=dji", "eval=kcomplexity", "methods=[clearance,greedy_arterial_buildable]",
         "block_ids=[[DJI.1_2_602]]", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "frontier_access.csv").read_text()
    assert "clearance" in table and "greedy_arterial_buildable" in table
    assert (tmp_path / "curve_access_DJI.1_2_602.png").exists()


def _terminal_benefit_by_method(csv_path: Path) -> dict[str, float]:
    """Benefit at the max road-density sample per method, read from a frontier CSV."""
    term: dict[str, tuple[float, float]] = {}
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            m = r["method"]
            rd, b = float(r["road_density_m_per_ha"]), float(r["benefit"])
            if m not in term or rd > term[m][0]:
                term[m] = (rd, b)
    return {m: b for m, (_, b) in term.items()}


def test_compare_two_adjacent_block_region_arterial_beats_clearance_directness(
    tmp_path: Path,
) -> None:
    # The multi-block region compare path: an adjacent DJI pair as ONE seed group, reblocked jointly
    # per method, curves keyed by "DJI.3_1_1808+DJI.3_1_1809". On directness the buildable-arterial
    # method (straight chords) reaches a higher terminal directness than clearance (least-cost) --
    # same reason it wins directness in the flagship examples.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "methods=[clearance,greedy_arterial_buildable]",
         "block_ids=[[DJI.3_1_1808,DJI.3_1_1809]]", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    label = "DJI.3_1_1808+DJI.3_1_1809"
    for metric in ("access", "efficiency", "directness"):
        assert (tmp_path / f"frontier_{metric}.csv").exists()
        assert (tmp_path / f"curve_{metric}_{label}.png").exists()
    term = _terminal_benefit_by_method(tmp_path / "frontier_directness.csv")
    assert term["greedy_arterial_buildable"] > term["clearance"]


def test_compare_report_writes_frontier(tmp_path: Path) -> None:
    from reblock.budget import Curve
    from reblock.compare import MethodCurve, compare_report
    results = [
        MethodCurve("clearance", "b1", "access", Curve([0.0, 1.0], [0.0, 0.9])),
        MethodCurve("topology", "b1", "access", Curve([0.0, 2.0], [0.0, 0.9])),
    ]
    compare_report(results, tmp_path, method_order=["clearance", "topology"])
    assert (tmp_path / "frontier_access.csv").exists()
    assert (tmp_path / "curve_access_b1.png").exists()


def test_frontier_csv_has_road_density_and_benefit_samples(tmp_path: Path) -> None:
    from reblock.budget import Curve
    from reblock.compare import MethodCurve
    from reblock.emit import compare_report
    c = Curve(cost=[0.0, 100.0], benefit=[0.0, 0.8])
    mc = MethodCurve("clearance", "B1", "access", c, pct_paved=0.041, pct_displaced=0.0)
    compare_report([mc], tmp_path, cost="length", method_order=["clearance"])
    with (tmp_path / "frontier_access.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"method", "block", "road_density_m_per_ha", "benefit"}
    # both sampled frontier points are present, in curve order
    assert [(r["road_density_m_per_ha"], r["benefit"]) for r in rows] == [
        ("0.0000", "0.0000"), ("100.0000", "0.8000")]


def test_compare_method_sweep_expands_over_param_values(tmp_path: Path) -> None:
    # method_sweep expands ONE base method over a param's values -> `{base}_{param}{value}` methods,
    # replacing hand-written all_methods entries. Here: clearance at repulsion -3/0/3, one plot.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity", "max_blocks=1",
         "methods=[]", "method_sweep={base: clearance, param: repulsion, values: [-3, 0, 3]}",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "frontier_directness.csv").read_text()
    assert "clearance_repulsion-3" in table
    assert "clearance_repulsion0" in table
    assert "clearance_repulsion3" in table
