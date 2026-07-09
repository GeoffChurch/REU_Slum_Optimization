import subprocess
import sys
from pathlib import Path
from typing import cast

import geopandas as gpd
from hydra import compose, initialize
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block, Result
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.topology import TopologyMethod
from reblock.run import RunConfig, run

PHULE = str(Path(__file__).resolve().parents[1] / "ext" / "topology" / "examples"
            / "data" / "phule_nagar_v6.shp")

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    """An n x n grid of unit-square parcels whose only street frontage is the
    outer boundary (mirrors tests/test_render.py's fixture of the same
    shape). Unlike the real fixtures below, this has genuine peel-depth
    signal: the centre parcel(s) sit two hops from the boundary.
    """
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="synthetic_3x3", crs=UTM, boundary=boundary,
                 parcels=parcels, streets=streets)


def test_cli_entrypoint_smoke(tmp_path: Path) -> None:
    # Exercises the real @hydra.main entrypoint (python -m reblock.run) against
    # the conf/ config groups, not just run(RunConfig(...)) directly -- catches
    # breakage in CLI arg parsing / config-group composition that calling run()
    # in-process can't see. hydra.run.dir is redirected to tmp_path so the
    # Hydra-created output dir (and the PNGs written under it) land outside
    # the repo tree instead of littering it on every test run.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.run",
         f"shapefile={PHULE}", "max_blocks=1", "assumed_crs=3857",
         "render.enabled=true", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "phule_0" in result.stdout
    assert "k_before" in result.stdout

    befores = list(tmp_path.glob("phule_0_before.png"))
    afters = list(tmp_path.glob("phule_0_*_after.png"))
    assert len(befores) == 1 and befores[0].stat().st_size > 0
    assert len(afters) >= 1 and afters[0].stat().st_size > 0


def test_cli_block_ids_renders_single_capetown_block(tmp_path: Path) -> None:
    # Validates the README recipe end-to-end through the real @hydra.main entrypoint:
    # block_ids builds ONLY the flagship, and render.enabled writes its before/after PNGs.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.run",
         "data=capetown", "method=peel", "eval=kcomplexity",
         "block_ids=[ZAF.9.3.1_1_44882]", "render.enabled=true",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ZAF.9.3.1_1_44882" in result.stdout

    befores = list(tmp_path.glob("ZAF.9.3.1_1_44882_before.png"))
    afters = list(tmp_path.glob("ZAF.9.3.1_1_44882_*_after.png"))
    assert len(befores) == 1 and befores[0].stat().st_size > 0
    assert len(afters) >= 1 and afters[0].stat().st_size > 0


def test_end_to_end_phule_wiring() -> None:
    # Wiring proof on real data: run() returns well-formed Results and writes
    # nothing (rendering is an emitter now, exercised by the CLI test below).
    results = run(RunConfig(shapefile=PHULE, region_id="phule", alpha=2.0, seed=0,
                            max_blocks=1, assumed_crs=3857))
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, Result)
    assert r.block.block_id == "phule_0"
    assert r.metric("kcomplexity", "k_after") <= r.metric("kcomplexity", "k_before")
    assert r.metric("kcomplexity", "delta_k") >= 0


def test_run_is_pure_deterministic_and_leaves_global_rng_untouched() -> None:
    import numpy as np
    cfg = RunConfig(shapefile=PHULE, region_id="phule", alpha=2.0, seed=0,
                    max_blocks=1, assumed_crs=3857)
    np.random.seed(777)
    state_before = np.random.get_state()[1].tolist()
    r1 = run(cfg)
    r2 = run(cfg)
    # no global RNG side-effect
    assert np.random.get_state()[1].tolist() == state_before
    # bit-identical repeats
    assert [x.proposal.proposal_id for x in r1] == [x.proposal.proposal_id for x in r2]
    assert (r1[0].metric("kcomplexity", "delta_k")
            == r2[0].metric("kcomplexity", "delta_k"))


def test_runconfig_accepts_explicit_data_method_eval_overrides() -> None:
    # RunConfig's flat fields (shapefile=, alpha=, ...) are sugar: __post_init__
    # only derives data/method/eval from them when those are left unset. A
    # caller can instead hand RunConfig the same _target_-bearing shapes Hydra
    # compose would produce -- e.g. to select WeakDualKEval, or to combine
    # more than one eval -- and run() must instantiate them identically either
    # way (single code path, no flat-field derivation involved here at all).
    cfg = RunConfig(
        max_blocks=1,
        data={"_target_": "reblock.data.shapefile.ShapefileSource",
              "path": PHULE, "region_id": "phule", "assumed_crs": 3857},
        method={"_target_": "reblock.methods.topology.TopologyMethod", "alpha": 2.0, "seed": 0},
        eval=[{"_target_": "reblock.eval.kcomplexity.KComplexityEval"},
              {"_target_": "reblock.eval.kcomplexity.WeakDualKEval"}],
    )

    results = run(cfg)

    assert len(results) == 1
    r = results[0]
    assert {m.eval for m in r.metrics} == {"kcomplexity", "weakdual_k"}
    assert r.metric("kcomplexity", "k_after") <= r.metric("kcomplexity", "k_before")


def test_hydra_compose_wires_config_groups() -> None:
    # Composes the real conf/ config groups (not a hand-built RunConfig), so a
    # break in defaults/_target_ wiring (e.g. a typo'd module path) fails here
    # instead of only surfacing in the CLI subprocess test.
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "data=phule", "method=topology", "eval=kcomplexity",
            f"shapefile={PHULE}", "assumed_crs=3857", "max_blocks=1",
        ])
        results = run(cfg)

    assert len(results) == 1
    r = results[0]
    assert r.metric("kcomplexity", "k_after") <= r.metric("kcomplexity", "k_before")


def test_hydra_compose_wires_peel_method() -> None:
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "data=phule", "method=peel", "eval=kcomplexity",
            f"shapefile={PHULE}", "assumed_crs=3857", "max_blocks=1",
        ])
        results = run(cfg)
    assert len(results) == 1
    r = results[0]
    assert r.proposal.method == "peel" and r.proposal.proposal_id == "peel_tol0.5"
    assert r.metric("kcomplexity", "k_after") <= r.metric("kcomplexity", "k_before")


def test_hydra_compose_wires_kblock_source_and_peel_pipeline() -> None:
    # First non-trivial real reblocking through the WHOLE pipeline: unlike Phule (every
    # block scores k=1, no peel signal -- see test_topology_reblocks_a_synthetic_nested_block),
    # kblock's Voronoi blocks have real interior parcels, so PeelReblocker demonstrably
    # improves some of them. max_blocks=1 already suffices: the first sorted DJI block
    # (DJI.1_2_602) improves under peel (delta_k=3), so it's kept at 1 for speed rather
    # than probing further.
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "data=dji", "method=peel", "eval=kcomplexity", "max_blocks=1",
        ])
        results = run(cfg)

    assert len(results) >= 1
    for r in results:
        kc = next(m for m in r.metrics if m.eval == "kcomplexity")
        assert "geometric_access_max_m" in kc.values
    assert max(r.metric("kcomplexity", "delta_k") for r in results) > 0


def test_block_ids_targets_one_capetown_block_through_the_pipeline() -> None:
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "data=capetown", "method=peel", "eval=kcomplexity",
            "block_ids=[ZAF.9.3.1_1_44882]", "max_blocks=10",
        ])
        results = run(cfg)
    # block_ids overrides the coarse max_blocks front-selection: exactly the one block.
    assert [r.block.block_id for r in results] == ["ZAF.9.3.1_1_44882"]
    r = results[0]
    assert r.metric("kcomplexity", "geometric_access_max_m") >= 0.0
    assert r.metric("kcomplexity", "delta_k") > 0   # peel flattens this deep block


def test_topology_reblocks_a_synthetic_nested_block() -> None:
    # Capstone efficacy proof. Both real fixtures available to this pipeline --
    # Phule Nagar (all 370/370 blocks) and ext/topology/Data/Epworth_demo.shp
    # (33-parcel single block; the other small real slum-block fixture in this
    # repo) -- score k_before == 1 for EVERY parcel under the honest BFS-peel
    # metric (reblock.derive.access.parcel_access_layers): every parcel
    # touches Block.streets, because Block.streets there is the whole
    # dissolved-union boundary of a fragmented block, which necessarily
    # includes the boundary ring of every inter-hut/inter-parcel gap as
    # "street" frontage. That is CORRECT behaviour of the honest peel metric
    # on this kind of fragmented real data -- not a bug to work around -- but
    # it means neither real fixture has any peel *efficacy* signal to
    # demonstrate (verified: Epworth_demo also gives k_before=1,
    # added_road_length_m=0, delta_k=0 under TopologyMethod(seed=0), same as
    # every Phule block). So the efficacy proof uses a synthetic 3x3 grid
    # block instead, direct-constructed (no Source): only the outer boundary
    # is a street, so the centre parcel is genuinely landlocked at
    # peel-depth 2, and TopologyMethod's greedy road-builder demonstrably
    # reaches it.
    block = _grid_block(3)
    proposal = TopologyMethod(alpha=2.0, seed=0).propose(block)
    m = KComplexityEval().score(block, proposal).values

    assert m["k_before"] == 2.0
    assert m["k_after"] < m["k_before"]
    assert m["delta_k"] > 0
    assert m["added_road_length_m"] > 0
    assert proposal.roads is not None and len(proposal.roads) > 0
