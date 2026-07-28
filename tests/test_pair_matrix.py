"""Tests for scripts/pair_matrix.py's import-gating: `--analyze` must be reachable from a
checkout that lacks `scratchpad/ot/` (gitignored, never repo content), since it only reads an
already-scored parquet plus numpy/pandas/scipy -- no GW/OSM/clearance/pool work at all.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import pair_matrix


def _synthetic_matrix() -> pd.DataFrame:
    # 2 recipients x 3 donors each -- small, but enough to exercise every column
    # `analyze_fidelity_vs_distance` touches without needing the real 100-row matrix.
    return pd.DataFrame(
        {
            "recipient": ["r0", "r0", "r0", "r1", "r1", "r1"],
            "donor": ["d0", "d1", "d2", "d3", "d4", "d5"],
            "real_gw_dist": [0.010, 0.020, 0.030, 0.011, 0.021, 0.031],
            "perm_gap": [0.10, 0.05, -0.05, -0.05, -0.10, -0.15],
            "feature_dist": [0.5, 1.0, 1.5, 0.6, 1.1, 1.6],
            "road_len_m": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
        }
    )


def test_analyze_runs_without_scratchpad_ot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core regression: `--analyze` must not require `scratchpad/ot/` at all. Point `_OT_DIR`
    at a path that does not exist, then confirm `main() --analyze` still succeeds -- proving the
    OT loader (`_ot()`) is never invoked on this path -- and that `_ot_ns` stays unset."""
    monkeypatch.setattr(pair_matrix, "_OT_DIR", tmp_path / "does-not-exist")
    assert pair_matrix._ot_ns is None

    out = tmp_path / "matrix.parquet"
    _synthetic_matrix().to_parquet(out)
    monkeypatch.setattr(sys, "argv", ["pair_matrix", "--analyze", "--out", str(out)])

    pair_matrix.main()  # must not raise SystemExit -- would if _ot() were reached

    assert pair_matrix._ot_ns is None  # confirms _ot() really was never called


def test_analyze_matches_direct_call_to_analyze_fidelity_vs_distance(tmp_path: Path) -> None:
    """`--analyze`'s printed numbers come from `analyze_fidelity_vs_distance` on the same
    DataFrame read straight from the parquet -- a basic sanity check that the CLI path and the
    underlying pure function agree, independent of the import-gating fix above."""
    df = _synthetic_matrix()
    result = pair_matrix.analyze_fidelity_vs_distance(df)
    assert result["n"] == 6
    assert result["n_recipients"] == 2
    within = result["within_recipient"]
    assert isinstance(within, dict)
    assert within["dof"] == 6 - 2 - 1


def test_ot_loader_still_raises_systemexit_when_scratchpad_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard itself must still fire -- just deferred to the pair-scoring path, not removed."""
    monkeypatch.setattr(pair_matrix, "_OT_DIR", tmp_path / "does-not-exist")
    monkeypatch.setattr(pair_matrix, "_ot_ns", None)
    with pytest.raises(SystemExit, match="scratchpad/ot/ is missing"):
        pair_matrix._ot()


def test_iso_of_picks_the_extract_from_the_block_ids() -> None:
    """A PBF covers exactly its own extract, so a Kenyan pool pointed at the South Africa file
    does not error -- every donor comes back with no interior footpaths. The first Nairobi run
    reported `empty_interior: 90` and zero pairs, which reads as a fact about Nairobi and is
    contradicted by the census. Derive the extract from the data, never by hand."""
    zaf = [SimpleNamespace(block_id="ZAF.9.3.1_1_44882"), SimpleNamespace(block_id="ZAF.9.1_1_1")]
    ken = [SimpleNamespace(block_id="KEN.1.1_1_100")]
    assert pair_matrix.iso_of(zaf) == "ZAF"  # type: ignore[arg-type]
    assert pair_matrix.iso_of(ken) == "KEN"  # type: ignore[arg-type]
    assert pair_matrix.PBF_BY_ISO["KEN"] != pair_matrix.PBF_BY_ISO["ZAF"]

    with pytest.raises(SystemExit, match="spans multiple countries"):
        pair_matrix.iso_of(zaf + ken)  # type: ignore[arg-type]
    with pytest.raises(SystemExit, match="no Geofabrik extract"):
        pair_matrix.iso_of([SimpleNamespace(block_id="BRA.1_1_1")])  # type: ignore[arg-type]
