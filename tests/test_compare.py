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
    table = (tmp_path / "auc_table.csv").read_text()
    assert "dijkstra" in table and "peel" in table
    assert list(tmp_path.glob("curve_*.png"))


def test_compare_report_writes(tmp_path: Path) -> None:
    from reblock.budget import Curve
    from reblock.compare import MethodCurve, compare_report
    results = [
        MethodCurve("dijkstra", "b1", Curve([0.0, 1.0], [0.0, 0.9]), 0.8),
        MethodCurve("peel", "b1", Curve([0.0, 2.0], [0.0, 0.9]), 0.5),
    ]
    compare_report(results, tmp_path)
    assert (tmp_path / "auc_table.csv").exists()
    assert (tmp_path / "curve_b1.png").exists()
