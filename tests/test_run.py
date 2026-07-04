import subprocess
import sys
from pathlib import Path

from reblock.data.shapefile import ShapefileSource
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.topology import TopologyMethod
from reblock.run import RunConfig, run

PHULE = str(Path(__file__).resolve().parents[1] / "ext" / "topology" / "examples"
            / "data" / "phule_nagar_v6.shp")


def test_cli_entrypoint_smoke(tmp_path: Path) -> None:
    # Exercises the real @hydra.main/ConfigStore entrypoint (python -m reblock.run),
    # not just run(RunConfig(...)) directly -- catches breakage in CLI arg parsing /
    # config registration that calling run() in-process can't see. hydra.run.dir is
    # redirected to tmp_path so the Hydra-created outputs/ dir lands outside the repo
    # tree instead of littering it on every test run (it's gitignored, but still).
    result = subprocess.run(
        [sys.executable, "-m", "reblock.run",
         f"shapefile={PHULE}", "max_blocks=1", "assumed_crs=3857", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "phule_0" in result.stdout
    assert "k_before" in result.stdout


def test_end_to_end_phule() -> None:
    results = run(RunConfig(shapefile=PHULE, region_id="phule", alpha=2.0, seed=0, max_blocks=1,
                             assumed_crs=3857))
    assert len(results) == 1
    v = results[0].values
    assert v["k_after"] <= v["k_before"] and v["delta_k"] >= 0


def test_pipeline_actually_reblocks_a_real_phule_block() -> None:
    # Capstone efficacy proof: phule_0 has no interior parcels (delta_k=0), so the
    # end-to-end test above proves wiring but never shows a real reblocking. Scan the
    # region for the FIRST block the full pipeline genuinely improves (adds road AND
    # drops k), and assert it. Deterministic under seed=0. The whole 370-block region
    # scans in <2s (build_all_roads ~5ms/block here), so no cap is needed; early-exit
    # on first success. Empirically the first (and only) improving block is phule_105.
    region = ShapefileSource(PHULE, region_id="phule", assumed_crs=3857).region()
    evaluator = KComplexityEval()

    scanned = 0
    found = None
    for block in region.blocks:
        scanned += 1
        proposal = TopologyMethod(seed=0).propose(block)
        m = evaluator.score(block, proposal).values
        if m["added_road_length_m"] > 0 and m["delta_k"] > 0:
            found = (block.block_id, m, proposal)
            break

    if found is None:
        raise AssertionError(
            f"scanned all {scanned} Phule Nagar blocks; the full pipeline never reblocked "
            f"real geometry (no block had added_road_length_m>0 AND delta_k>0). The slice "
            f"wires together but demonstrates no efficacy on this dataset -- investigate."
        )

    block_id, m, proposal = found
    assert m["k_after"] < m["k_before"], (block_id, m)
    assert m["delta_k"] > 0 and m["added_road_length_m"] > 0
    assert proposal.roads is not None and len(proposal.roads) > 0
