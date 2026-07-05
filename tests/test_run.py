import subprocess
import sys
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
from hydra import compose, initialize
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block, Metrics, Proposal, Result
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.topology import TopologyMethod
from reblock.run import RunConfig, _render_block, run

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


def test_render_after_filenames_stay_unique_when_proposal_id_is_empty(tmp_path: Path) -> None:
    # Guard: the after-PNG name is `{block_id}_{proposal_id}_after.png`, and
    # `proposal_id` defaults to "" -- so two proposals that both leave it empty
    # (a future method might) must NOT collide onto one filename and silently
    # overwrite. run() falls back to a per-proposal index. No current Method
    # produces an empty proposal_id, so this exercises the render helper
    # directly with hand-built empty-id proposals.
    block = _grid_block(3)
    layers = pd.Series(
        [1] * len(block.parcels),
        index=pd.Index(block.parcels["parcel_id"], name="parcel_id"),
    )

    def _kc() -> Metrics:
        return Metrics(block_id=block.block_id, method="x", eval="kcomplexity",
                       values={"delta_k": 0.0},
                       fields={"access_before": layers, "access_after": layers})

    per_proposal: list[tuple[Proposal, tuple[Metrics, ...]]] = [
        (Proposal(block_id=block.block_id, crs=UTM, proposal_id=""), (_kc(),)),
        (Proposal(block_id=block.block_id, crs=UTM, proposal_id=""), (_kc(),)),
    ]
    _render_block(block, per_proposal, tmp_path)

    afters = sorted(p.name for p in tmp_path.glob("*_after.png"))
    assert afters == ["synthetic_3x3_proposal0_after.png",
                      "synthetic_3x3_proposal1_after.png"]


def test_cli_entrypoint_smoke(tmp_path: Path) -> None:
    # Exercises the real @hydra.main entrypoint (python -m reblock.run) against
    # the conf/ config groups, not just run(RunConfig(...)) directly -- catches
    # breakage in CLI arg parsing / config-group composition that calling run()
    # in-process can't see. hydra.run.dir is redirected to tmp_path so the
    # Hydra-created output dir (and the renders/ written under it) land outside
    # the repo tree instead of littering it on every test run.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.run",
         f"shapefile={PHULE}", "max_blocks=1", "assumed_crs=3857",
         "render_dir=renders", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "phule_0" in result.stdout
    assert "k_before" in result.stdout

    befores = list(tmp_path.glob("renders/phule_0_before.png"))
    afters = list(tmp_path.glob("renders/phule_0_*_after.png"))
    assert len(befores) == 1 and befores[0].stat().st_size > 0
    assert len(afters) >= 1 and afters[0].stat().st_size > 0


def test_end_to_end_phule_wiring(tmp_path: Path) -> None:
    # Wiring proof on real data: phule_0 has no interior parcels reachable by
    # the greedy road-builder (see the efficacy test below for why NO Phule or
    # Epworth_demo block shows peel-metric improvement), so this asserts the
    # pipeline produces well-formed Results and renders -- not that it
    # improves anything on this particular block.
    results = run(
        RunConfig(shapefile=PHULE, region_id="phule", alpha=2.0, seed=0, max_blocks=1,
                  assumed_crs=3857, render_dir="renders"),
        render_base=tmp_path,
    )

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, Result)
    assert r.block.block_id == "phule_0"
    assert r.metric("kcomplexity", "k_after") <= r.metric("kcomplexity", "k_before")
    assert r.metric("kcomplexity", "delta_k") >= 0

    before_png = tmp_path / "renders" / "phule_0_before.png"
    after_png = tmp_path / "renders" / f"phule_0_{r.proposal.proposal_id}_after.png"
    assert before_png.exists() and before_png.stat().st_size > 0
    assert after_png.exists() and after_png.stat().st_size > 0


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
        method=[{"_target_": "reblock.methods.topology.TopologyMethod", "alpha": 2.0, "seed": 0}],
        eval=[{"_target_": "reblock.eval.kcomplexity.KComplexityEval"},
              {"_target_": "reblock.eval.kcomplexity.WeakDualKEval"}],
    )

    results = run(cfg)

    assert len(results) == 1
    r = results[0]
    assert {m.eval for m in r.metrics} == {"kcomplexity", "weakdual_k"}
    assert r.metric("kcomplexity", "k_after") <= r.metric("kcomplexity", "k_before")


def test_run_scores_multiple_methods_in_one_call(tmp_path: Path) -> None:
    # Regression guard for the render design's "one before, N afters" premise:
    # run() must score EVERY configured method against each block and bundle
    # one Result per (block, method). Mirrors
    # test_runconfig_accepts_explicit_data_method_eval_overrides, but flips it
    # to TWO methods (distinct seeds -> distinct proposal_ids) x one eval,
    # instead of one method x two evals. Uses the same fast real phule_0 block
    # (max_blocks=1) as that test -- run() instantiates its source via
    # hydra.utils.instantiate(cfg.data), which needs a _target_-referenceable
    # class, so a direct-constructed synthetic Block can't be fed through the
    # real run() path; phule_0 is the fast, already-loaded real block.
    cfg = RunConfig(
        max_blocks=1,
        render_dir="renders",
        data={"_target_": "reblock.data.shapefile.ShapefileSource",
              "path": PHULE, "region_id": "phule", "assumed_crs": 3857},
        method=[
            {"_target_": "reblock.methods.topology.TopologyMethod", "alpha": 2.0, "seed": 0},
            {"_target_": "reblock.methods.topology.TopologyMethod", "alpha": 2.0, "seed": 1},
        ],
        eval=[{"_target_": "reblock.eval.kcomplexity.KComplexityEval"}],
    )

    results = run(cfg, render_base=tmp_path)

    # Two methods, one block -> two Results (one per method), same block.
    assert len(results) == 2
    assert results[0].block.block_id == results[1].block.block_id == "phule_0"
    ids = [r.proposal.proposal_id for r in results]
    assert ids == ["topology_a2.0_s0", "topology_a2.0_s1"]
    assert len(set(ids)) == 2  # distinct proposals, not the same one twice

    # Render layout: exactly ONE before per block, and one distinct after per
    # proposal (the "one before, N afters" the shared-vmax design depends on).
    renders = tmp_path / "renders"
    befores = list(renders.glob("phule_0_before.png"))
    afters = sorted(renders.glob("phule_0_*_after.png"))
    assert len(befores) == 1 and befores[0].stat().st_size > 0
    assert [p.name for p in afters] == [
        "phule_0_topology_a2.0_s0_after.png",
        "phule_0_topology_a2.0_s1_after.png",
    ]
    assert all(p.stat().st_size > 0 for p in afters)


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
