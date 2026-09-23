"""The consensus study: predict a block's real footpaths from its mapped neighbours, and ask what
the prediction is worth as a reblocker -- every arm an ordinary Method, every one scored through the
SAME truncations.

ARMS are Methods built from conf/ like any other (`conf/consensus_matrix.yaml`), each named by the
run-config overrides that select it:

- the consensus: `demand_greedy` routed toward a weighted field of GW-transported donor footpaths
  (`method=consensus`, `reblock.transplant.consensus`);
- the single-donor transplant: the closest donor's footpaths, routed onto the recipient's gaps
  (`method=donor_transplant`);
- a from-scratch baseline (`method=clearance`);
- the block's own mapped footpaths (`method=osm_footpaths desire_source=pool`), which is also the
  REFERENCE every prediction is scored against.

The donor arms come in pairs, `donors=held_out` (donors > 2 km away) and `donors=leaky` (nearest
wherever); the gap between them is the leakage estimate, and the held-out arm is the one that can
be believed. `k_sweep` runs named arms again at each k (`donors.k`); a GW fit does not depend on k,
so the rungs share every solve through the derivation cache and nest.

TRUNCATION. No arm is scored on its own untruncated network, and every arm goes through one
function, `truncate`:

- Lens A and Lens B are `reblock.compare.lens_prefixes`, the truncation `scripts/compare_budgets`
  reports every shipped method at: the first street-first prefix displacing >= the universal
  `matched_displacement` (its permeability read off), and the first reaching >= the universal
  `matched_permeability` (the displacement it cost read off).
- The PREDICTION question -- did the arm put paths where the real ones are
  (`reblock.eval.agreement`) -- is asked at ONE canonical truncation,
  `budget.prefix_to_displacement` at the displacement of the block's own network: the function
  Lens A uses. Displacement, not length, because it is the cost the metric reports, and a metre
  along a gap is not a metre through homes; matched to the real network's own so that "the same
  budget" is what the real network actually spent. Every arm, the reference included, overshoots
  it by at most one road, under the same rule. The reference it is scored AGAINST is the reference
  arm's whole network: the ground truth is not truncated.

Before this, the study matched budgets with its own code, and the copies disagreed: the consensus
took the shortest street-first prefix REACHING the reference's length, the baseline the longest
construction-order prefix WITHIN it, the consensus extracted at depth 1 and the baseline at 2, and
the sweep's single-donor arm was not matched at all.

    pixi run python -m scripts.consensus_matrix out=<parquet> recipients=4        # pilot
    pixi run python -m scripts.consensus_matrix out=<parquet> recipients=2 \\
        'run=[consensus_held_out,single_held_out,clearance,own]' \\
        'k_sweep.arms=[consensus_held_out,single_held_out]' 'k_sweep.ks=[1,3,8,15]'
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import pandas as pd
from geopandas import GeoDataFrame
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from reblock.budget import prefix_to_displacement
from reblock.compare import (
    LensPrefixes,
    PermeabilityConfig,
    lens_prefixes,
    load_permeability_config,
)
from reblock.contracts import Block, Method
from reblock.data.pools import DonorPool, ScreenedPool, evenly_spaced
from reblock.derivations import propose
from reblock.emit import pct_displaced
from reblock.eval.agreement import buffered_iou, directional_chamfer
from reblock.permeability import EgressContext, permeability
from reblock.presets import load_method, load_research
from reblock.transplant.donors import TooFewDonors

CONF = Path("conf")


class StudyRow(TypedDict):
    """One (recipient, arm) row, exactly as written to the parquet."""

    recipient: str
    arm: str
    parcels: int
    params: str                     # the arm's Proposal.params, as JSON
    # Lens A: the first prefix displacing >= matched_displacement
    a_road_m: float
    a_displacement: float
    a_permeability: float
    a_at_budget: bool               # False: the whole network displaces less than the budget
    # Lens B: the first prefix reaching >= matched_permeability
    b_road_m: float
    b_displacement: float
    b_permeability: float
    b_reached: bool
    # Prediction: the first prefix displacing >= the reference network's own displacement
    p_target_displacement: float
    p_road_m: float
    p_displacement: float
    p_permeability: float
    iou_3m: float
    iou_10m: float
    chamfer_precision_m: float      # mean distance from a predicted path to the nearest real one
    chamfer_recall_m: float         # mean distance from a real path to the nearest predicted one


class SkipRow(TypedDict):
    """A (recipient, arm) that proposed nothing to score, and why -- written beside the parquet."""

    recipient: str
    arm: str
    reason: str


@dataclass(frozen=True, eq=False)
class Truncations:
    """Every truncation an arm is scored at."""

    lenses: LensPrefixes
    prediction: GeoDataFrame


def truncate(ctx: EgressContext, roads: GeoDataFrame, target_displacement: float,
             pcfg: PermeabilityConfig) -> Truncations:
    """THE truncation every arm is scored through, and the only one."""
    return Truncations(lenses=lens_prefixes(ctx, roads, pcfg),
                       prediction=prefix_to_displacement(ctx.block, roads, target_displacement))


def _road_m(roads: GeoDataFrame) -> float:
    return float(roads.geometry.length.sum()) if len(roads) else 0.0


def score_recipient(block: Block, roads: Mapping[str, GeoDataFrame],
                    params: Mapping[str, Mapping[str, object]], reference: str,
                    pcfg: PermeabilityConfig) -> list[StudyRow]:
    """Every arm's row on one recipient. Nothing here reads an arm's roads except through
    `truncate`; the reference's whole network is only ever the thing agreement is measured
    against."""
    ctx = EgressContext.of(block, pcfg.params)
    truth = roads[reference]
    target = pct_displaced(truth, block.buildings)
    rows: list[StudyRow] = []
    for arm, arm_roads in roads.items():
        t = truncate(ctx, arm_roads, target, pcfg)
        a, b, p = t.lenses.displacement, t.lenses.permeability, t.prediction
        a_disp = pct_displaced(a, block.buildings)
        precision_m, recall_m = directional_chamfer(p, truth)
        rows.append(StudyRow(
            recipient=block.block_id, arm=arm, parcels=len(block.parcels),
            params=json.dumps(dict(params[arm]), default=str, sort_keys=True),
            a_road_m=_road_m(a), a_displacement=a_disp, a_permeability=permeability(ctx, a),
            # `compare_budgets`' own reading of "at the budget" (its `OutcomeRow.at_budget`).
            a_at_budget=a_disp >= pcfg.matched_displacement - 1e-9,
            b_road_m=_road_m(b), b_displacement=pct_displaced(b, block.buildings),
            b_permeability=permeability(ctx, b), b_reached=t.lenses.reached,
            p_target_displacement=target, p_road_m=_road_m(p),
            p_displacement=pct_displaced(p, block.buildings), p_permeability=permeability(ctx, p),
            # IoU at two radii because buffers stop overlapping past 2r -- at 3 m it reads 0 for
            # anything more than 6 m off, which a predicted network easily is. Chamfer is kept
            # DIRECTIONAL, per its own contract: blending the two hides which way a prediction
            # fails.
            iou_3m=buffered_iou(p, truth, r=3.0), iou_10m=buffered_iou(p, truth, r=10.0),
            chamfer_precision_m=precision_m, chamfer_recall_m=recall_m))
    return rows


def _run_config(overrides: Sequence[str]) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONF.resolve())):
        return compose(config_name="config", overrides=list(overrides))


@dataclass(frozen=True, eq=False)
class Study:
    """The study's configuration, resolved: the pool recipients come from, and the arms."""

    pool: DonorPool
    arms: dict[str, Method]
    reference: str
    recipients: int
    out: Path


def arm_overrides(cfg: DictConfig) -> dict[str, list[str]]:
    """Each selected arm's run-config overrides -- `common` first -- with every `k_sweep` rung
    added as `{arm}_k{k}`, which appends `donors.k={k}`."""
    common = [str(o) for o in cfg.common]
    out = {str(name): [*common, *(str(o) for o in cfg.arms[name])] for name in cfg.run}
    for name in cfg.k_sweep.arms:
        for k in cfg.k_sweep.ks:
            out[f"{name}_k{k}"] = [*common, *(str(o) for o in cfg.arms[name]), f"donors.k={k}"]
    return out


def load_study(cfg: DictConfig) -> Study:
    if cfg.reference not in cfg.run:
        raise ValueError(f"reference arm {cfg.reference!r} must be run: every prediction is "
                         f"scored against its network")
    # The pool as the donor arms draw it: `common` selects it for every arm alike, and arms
    # built from an equal pool share its one materialization.
    pool = load_research(_run_config([str(o) for o in cfg.common]).donor_pool, ScreenedPool)
    arms = {name: load_method(_run_config(o).method) for name, o in arm_overrides(cfg).items()}
    return Study(pool=pool, arms=arms, reference=str(cfg.reference),
                 recipients=int(cfg.recipients), out=Path(str(cfg.out)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    with initialize_config_dir(version_base=None, config_dir=str(CONF.resolve())):
        cfg = compose(config_name="consensus_matrix", overrides=sys.argv[1:])
    study = load_study(cfg)
    pcfg = load_permeability_config(CONF)
    pools = study.pool.pools()

    # A recipient needs its OWN footpaths as ground truth, so it must itself be donatable.
    usable = sorted(set(pools.recipients) & set(pools.donors))
    chosen = evenly_spaced(usable, [float(len(b.parcels)) for b in pools.blocks],
                           study.recipients)
    print(f"  {len(usable):,} recipients with own OSM; running {len(chosen)} x "
          f"{len(study.arms)} arms", flush=True)

    rows: list[StudyRow] = []
    skips: list[SkipRow] = []
    done: set[str] = set()
    skips_path = study.out.with_suffix(".skips.json")
    if study.out.exists():
        # This script's own output, so its records are the `StudyRow`s and `SkipRow`s it wrote.
        rows = cast(list[StudyRow], pd.read_parquet(study.out).to_dict("records"))
        skips = cast(list[SkipRow], json.loads(skips_path.read_text()))
        done = {r["recipient"] for r in rows}
        print(f"resuming from {study.out}: {len(done)} recipients", flush=True)

    for n, i in enumerate(chosen, 1):
        block = pools.blocks[i]
        if block.block_id in done:
            continue
        t0 = time.time()
        roads: dict[str, GeoDataFrame] = {}
        params: dict[str, Mapping[str, object]] = {}
        for arm, method in study.arms.items():
            try:
                proposal = propose(method, block)
            except TooFewDonors as exc:
                skips.append(SkipRow(recipient=block.block_id, arm=arm, reason=str(exc)))
                continue
            assert proposal.roads is not None, f"{arm} proposed no road frame"
            roads[arm], params[arm] = proposal.roads, proposal.params
        if roads[study.reference].empty:
            skips.append(SkipRow(recipient=block.block_id, arm=study.reference,
                                 reason="no interior footpaths to score against"))
            continue
        rows += score_recipient(block, roads, params, study.reference, pcfg)
        print(f"  [{n}/{len(chosen)}] {block.block_id}: {len(roads)} arms "
              f"[{time.time() - t0:.1f}s]", flush=True)
        study.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(study.out)
        skips_path.write_text(json.dumps(skips, indent=1))

    df = pd.DataFrame(rows)
    print(f"\nwrote {study.out} ({len(df)} rows, {len(skips)} skipped)")
    if len(df):
        print(f"  medians -- Lens A at D={pcfg.matched_displacement:g}, Lens B at "
              f"P*={pcfg.matched_permeability:g}, prediction at the reference's displacement")
        print(f"  {'arm':28s} {'n':>3} {'A perm':>7} {'B disp':>7} {'B reached':>9} "
              f"{'P perm':>7} {'IoU@10m':>8} {'recall m':>9}")
        for name, g in df.groupby("arm", sort=False):
            print(f"  {name!s:28s} {len(g):>3} {g.a_permeability.median():>7.3f} "
                  f"{g.b_displacement.median():>7.3f} {g.b_reached.mean():>9.2f} "
                  f"{g.p_permeability.median():>7.3f} {g.iou_10m.median():>8.3f} "
                  f"{g.chamfer_recall_m.median():>9.1f}")


if __name__ == "__main__":
    main()
