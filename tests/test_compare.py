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
    table = (tmp_path / "frontier_external_connectivity.csv").read_text()
    assert "clearance" in table and "greedy_arterial_buildable" in table
    assert list(tmp_path.glob("curve_external_connectivity_*.png"))


def test_compare_displacement_metric_runs_and_writes_curves(tmp_path: Path) -> None:
    # displacement rides the ordinary MethodCurve machinery as a metric="displacement" row --
    # every compare() run grades it automatically (no cost= flag), alongside the two benefit
    # frontiers. clearance (fast) proves the wiring end-to-end; the axis arithmetic itself is
    # unit-tested in test_budget.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "max_blocks=1", "methods=[clearance]", "corridor_m=3.0",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    # displacement is a rising cost, reported separately from the length frontier -- never
    # inverted, and never a tradeoff table (that path is gone).
    assert (tmp_path / "displacement_vs_length.csv").exists()
    assert (tmp_path / "displacement_table.csv").exists()
    assert not list(tmp_path.glob("tradeoff_table_*.csv"))
    assert (tmp_path / "frontier_internal_connectivity.csv").exists()
    assert list(tmp_path.glob("displacement_*.png"))


def test_compare_emits_per_metric_frontier_tables(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "max_blocks=1", "methods=[clearance,greedy_arterial_buildable]",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    for metric in ("external_connectivity", "internal_connectivity"):
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
    table = (tmp_path / "frontier_external_connectivity.csv").read_text()
    assert "clearance" in table and "greedy_arterial_buildable" in table
    assert (tmp_path / "curve_external_connectivity_DJI.1_2_602.png").exists()


def _terminal_benefit_by_method(csv_path: Path) -> dict[str, float]:
    """Benefit at the max road-length sample per method, read from a frontier CSV."""
    term: dict[str, tuple[float, float]] = {}
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            m = r["method"]
            rd, b = float(r["road_length_m"]), float(r["benefit"])
            if m not in term or rd > term[m][0]:
                term[m] = (rd, b)
    return {m: b for m, (_, b) in term.items()}


def test_compare_two_adjacent_block_region_arterial_beats_clearance_internal_connectivity(
    tmp_path: Path,
) -> None:
    # The multi-block region compare path: an adjacent DJI pair as ONE seed group, reblocked
    # jointly per method, curves keyed by "DJI.3_1_1808+DJI.3_1_1809". clearance adds no roads at
    # all on this fixture (0 m, benefit 0.000 on every metric); the buildable-arterial method's
    # straight chords create real interior loops, so it reaches strictly higher terminal internal
    # connectivity.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "methods=[clearance,greedy_arterial_buildable]",
         "block_ids=[[DJI.3_1_1808,DJI.3_1_1809]]", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    label = "DJI.3_1_1808+DJI.3_1_1809"
    for metric in ("external_connectivity", "internal_connectivity"):
        assert (tmp_path / f"frontier_{metric}.csv").exists()
        assert (tmp_path / f"curve_{metric}_{label}.png").exists()
    term = _terminal_benefit_by_method(tmp_path / "frontier_internal_connectivity.csv")
    assert term["greedy_arterial_buildable"] > term["clearance"]


def test_compare_report_writes_frontier(tmp_path: Path) -> None:
    from reblock.budget import Curve
    from reblock.compare import MethodCurve, compare_report
    results = [
        MethodCurve("clearance", "b1", "external_connectivity", Curve([0.0, 1.0], [0.0, 0.9])),
        MethodCurve("topology", "b1", "external_connectivity", Curve([0.0, 2.0], [0.0, 0.9])),
    ]
    compare_report(results, tmp_path, method_order=["clearance", "topology"])
    assert (tmp_path / "frontier_external_connectivity.csv").exists()
    assert (tmp_path / "curve_external_connectivity_b1.png").exists()


def test_frontier_csv_has_road_length_and_benefit_samples(tmp_path: Path) -> None:
    from reblock.budget import Curve
    from reblock.compare import MethodCurve
    from reblock.emit import compare_report
    c = Curve(cost=[0.0, 100.0], benefit=[0.0, 0.8])
    mc = MethodCurve("clearance", "B1", "external_connectivity", c,
                     pct_paved=0.041, pct_displaced=0.0)
    compare_report([mc], tmp_path, method_order=["clearance"])
    with (tmp_path / "frontier_external_connectivity.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"method", "block", "road_length_m", "benefit"}
    # both sampled frontier points are present, in curve order
    assert [(r["road_length_m"], r["benefit"]) for r in rows] == [
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
    table = (tmp_path / "frontier_external_connectivity.csv").read_text()
    assert "clearance_repulsion-3" in table
    assert "clearance_repulsion0" in table
    assert "clearance_repulsion3" in table
