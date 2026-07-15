import csv
import subprocess
import sys
from pathlib import Path


def test_compare_writes_table_and_curves(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare",
         "data=dji", "eval=kcomplexity", "max_blocks=1",
         "methods=[dijkstra,peel]", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "auc_table_access.csv").read_text()
    assert "dijkstra" in table and "peel" in table
    assert list(tmp_path.glob("curve_access_*.png"))


def test_compare_displacement_cost_axis_runs_and_writes_curves(tmp_path: Path) -> None:
    # The headline recipe: cost=displacement grades every method on the buildings-displaced x-axis
    # (frontage methods land near 0). dijkstra (fast) proves the axis is reachable end-to-end; the
    # axis arithmetic itself is unit-tested in test_budget, and greedy_arterial_displacement is the
    # slow flagship. Before the whole-branch-review fix, cost was hardcoded to "length" and this
    # axis was unreachable from any recipe.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "max_blocks=1", "methods=[dijkstra]", "cost=displacement", "corridor_m=3.0",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    # displacement uses a tradeoff table (terminal benefit + buildings displaced), NOT the AUC
    # table -- AUC inverts on the displacement axis (a home-sparing method scores 0).
    assert (tmp_path / "tradeoff_table_directness.csv").exists()
    assert not (tmp_path / "auc_table_directness.csv").exists()
    assert "buildings_displaced" in (tmp_path / "tradeoff_table_directness.csv").read_text()
    assert list(tmp_path.glob("curve_directness_*.png"))


def test_compare_emits_per_metric_tables(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "max_blocks=1", "methods=[dijkstra,mesh]", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    for metric in ("access", "efficiency", "directness"):
        assert (tmp_path / f"auc_table_{metric}.csv").exists()
        assert list(tmp_path.glob(f"curve_{metric}_*.png"))


def test_compare_singleton_via_explicit_block_ids_matches_plain_single_block(
    tmp_path: Path,
) -> None:
    # An explicit list-of-lists with ONE singleton group takes build_regions's
    # region_builder-expansion branch (not the classic screen=identity/None fallback that
    # test_compare_writes_table_and_curves exercises) -- but a singleton region is still the
    # EXACT pre-region single-block path, so it must write out identically (keyed by the
    # plain block_id, no "+", no region jointness).
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare",
         "data=dji", "eval=kcomplexity", "methods=[dijkstra,peel]",
         "block_ids=[[DJI.1_2_602]]", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "auc_table_access.csv").read_text()
    assert "dijkstra" in table and "peel" in table
    assert (tmp_path / "curve_access_DJI.1_2_602.png").exists()


def test_compare_two_adjacent_block_region_writes_per_metric_and_arterial_beats_dijkstra(
    tmp_path: Path,
) -> None:
    # The multi-block region compare path (README recipe): an adjacent DJI pair as ONE seed
    # group is reblocked jointly per method (region_reblock), each method's 3-lens curves
    # keyed by the region label "DJI.3_1_1808+DJI.3_1_1809". On directness specifically, the
    # buildable-arterial method should beat dijkstra for the region -- same reason it does
    # per-block: it targets straight chords instead of a greedy shortest-path tree.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "methods=[dijkstra,greedy_arterial_buildable]",
         "block_ids=[[DJI.3_1_1808,DJI.3_1_1809]]", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    label = "DJI.3_1_1808+DJI.3_1_1809"
    for metric in ("access", "efficiency", "directness"):
        assert (tmp_path / f"auc_table_{metric}.csv").exists()
        assert (tmp_path / f"curve_{metric}_{label}.png").exists()

    with (tmp_path / "auc_table_directness.csv").open(newline="") as f:
        rows = {r["method"]: float(r["mean_auc"]) for r in csv.DictReader(f)}
    assert rows["greedy_arterial_buildable"] > rows["dijkstra"]


def test_compare_report_writes(tmp_path: Path) -> None:
    from reblock.budget import Curve
    from reblock.compare import MethodCurve, compare_report
    results = [
        MethodCurve("dijkstra", "b1", "access", Curve([0.0, 1.0], [0.0, 0.9]), 0.8),
        MethodCurve("peel", "b1", "access", Curve([0.0, 2.0], [0.0, 0.9]), 0.5),
    ]
    compare_report(results, tmp_path)
    assert (tmp_path / "auc_table_access.csv").exists()
    assert (tmp_path / "curve_access_b1.png").exists()


def test_auc_table_has_mean_pct_paved_column(tmp_path: Path) -> None:
    from reblock.budget import Curve
    from reblock.compare import MethodCurve
    from reblock.emit import compare_report
    c = Curve(cost=[0.0, 100.0], benefit=[0.0, 0.8])
    mc = MethodCurve("dijkstra", "B1", "access", c, 0.5, pct_paved=0.041, pct_displaced=0.0)
    compare_report([mc], tmp_path, cost="length")
    with (tmp_path / "auc_table_access.csv").open() as f:
        header = next(csv.reader(f))
    assert "mean_pct_paved" in header


def test_compare_method_sweep_expands_over_param_values(tmp_path: Path) -> None:
    # method_sweep expands ONE base method over a param's values -> `{base}_{param}{value}` methods,
    # replacing hand-written all_methods entries. Here: clearance at repulsion -3/0/3, one plot.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity", "max_blocks=1",
         "methods=[]", "method_sweep={base: clearance, param: repulsion, values: [-3, 0, 3]}",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "auc_table_directness.csv").read_text()
    assert "clearance_repulsion-3" in table
    assert "clearance_repulsion0" in table
    assert "clearance_repulsion3" in table
