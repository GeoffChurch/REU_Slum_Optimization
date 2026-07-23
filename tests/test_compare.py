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
    table = (tmp_path / "frontier_permeability.csv").read_text()
    assert "clearance" in table and "greedy_arterial_buildable" in table
    assert list(tmp_path.glob("frontier_*.png"))


def test_compare_displacement_metric_runs_and_writes_curves(tmp_path: Path) -> None:
    # displacement rides the ordinary MethodCurve machinery as a metric="displacement" row, used
    # only to re-base the permeability frontier's x-axis (emit.compare_report) -- every compare()
    # run grades it automatically (no cost= flag). clearance (fast) proves the wiring end-to-end;
    # the axis arithmetic itself is unit-tested in test_budget/test_permeability.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "max_blocks=1", "methods=[clearance]", "corridor_m=3.0",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    # ONE frontier now (permeability vs displacement) -- the retired three-curve/tradeoff surface
    # (per-metric frontier CSVs, a separate displacement table/curve) is gone.
    assert (tmp_path / "frontier_permeability.csv").exists()
    text = (tmp_path / "frontier_permeability.csv").read_text()
    assert "displacement" in text and "permeability" in text
    assert not (tmp_path / "displacement_vs_length.csv").exists()
    assert not (tmp_path / "displacement_table.csv").exists()
    assert not (tmp_path / "frontier_external_connectivity.csv").exists()
    assert not (tmp_path / "frontier_internal_connectivity.csv").exists()
    assert not list(tmp_path.glob("tradeoff_table_*.csv"))
    assert not list(tmp_path.glob("displacement_*.png"))
    assert not list(tmp_path.glob("curve_*.png"))


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
    table = (tmp_path / "frontier_permeability.csv").read_text()
    assert "clearance" in table and "greedy_arterial_buildable" in table
    assert (tmp_path / "frontier_DJI.1_2_602.png").exists()


def test_compare_report_writes_frontier(tmp_path: Path) -> None:
    from reblock.budget import Curve
    from reblock.compare import MethodCurve, compare_report
    results = [
        MethodCurve("clearance", "b1", "permeability", Curve([0.0, 1.0], [0.0, 0.9])),
        MethodCurve("topology", "b1", "permeability", Curve([0.0, 2.0], [0.0, 0.9])),
    ]
    compare_report(results, tmp_path, method_order=["clearance", "topology"],
                   matched_displacement=0.10, matched_permeability=0.60)
    assert (tmp_path / "frontier_permeability.csv").exists()
    assert (tmp_path / "frontier_b1.png").exists()


def test_frontier_csv_has_displacement_and_permeability_samples(tmp_path: Path) -> None:
    from reblock.budget import Curve
    from reblock.compare import MethodCurve
    from reblock.emit import compare_report
    c = Curve(cost=[0.0, 100.0], benefit=[0.0, 0.8])
    mc = MethodCurve("clearance", "B1", "permeability", c, pct_paved=0.041, pct_displaced=0.0)
    compare_report([mc], tmp_path, method_order=["clearance"],
                   matched_displacement=0.10, matched_permeability=0.60)
    with (tmp_path / "frontier_permeability.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"method", "block", "displacement", "permeability"}
    # both sampled frontier points are present, in curve order (permeability is %.6g -- see
    # emit.py -- so small ratio values don't round to 0); no displacement row was supplied, so the
    # x-axis falls back to the permeability curve's own cost (cumulative road length).
    assert [(r["displacement"], r["permeability"]) for r in rows] == [
        ("0.0000", "0"), ("100.0000", "0.8")]


def test_compare_method_sweep_expands_over_param_values(tmp_path: Path) -> None:
    # method_sweep expands ONE base method over a param's values -> `{base}_{param}{value}` methods,
    # replacing hand-written all_methods entries. Here: clearance at repulsion -3/0/3, one plot.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity", "max_blocks=1",
         "methods=[]", "method_sweep={base: clearance, param: repulsion, values: [-3, 0, 3]}",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "frontier_permeability.csv").read_text()
    assert "clearance_repulsion-3" in table
    assert "clearance_repulsion0" in table
    assert "clearance_repulsion3" in table
