"""Two-lens method comparison for the multiblock example (replaces render_methods_matched.py).

Reblocks each method once over the region (timed), then reports it under two budgets:

  Lens A -- fixed OUTCOME (depth target D): the drainage-ordered road prefix that first brings
    every parcel within access-depth <= D (`budget.prefix_to_depth`); reports the road length,
    displacement and wall-clock propose time it took. A fixed input that never reaches D
    (osm_footpaths) is reported unreached with its floor depth.

  Lens B -- fixed COST (matched road budget): every method truncated to the sparsest method's
    total added road length (`budget.matched_budget` + `truncate_to_length`); reports benefit on
    both axes (external + internal connectivity) + displacement.

Both lenses render one after-heatmap per method (Lens A at the depth-D prefix, Lens B at the
matched budget), re-scoring access-depth on the truncated roads exactly as render_methods_matched
did.

Run (module form -- mirrors scripts/render_methods_matched.py's Hydra bootstrapping):
  pixi run python -m scripts.compare_budgets <out_dir> <target_depth> <m1,m2,...> \
       <hydra override>...

  e.g. examples/multiblock 3 clearance,greedy_arterial_buildable,osm_footpaths \
       data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
       block_ids=[[ZAF.9.3.1_1_5810]] all_methods.clearance.max_roads=3000 \
       all_methods.clearance.depth_target=3 \
       all_methods.greedy_arterial_buildable.candidate_policy=fixed \
       +all_methods.greedy_arterial_buildable.max_anchors=64 \
       desire_source.snapshot=examples/multiblock/desire_lines_5810.geojson
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from reblock.budget import (
    access_benefit,
    building_radii,
    commute_ratio,
    displacement,
    matched_budget,
    prefix_to_depth,
    truncate_to_length,
)
from reblock.contracts import Block, Method, Proposal, Screen, Source
from reblock.emit import _displaced_points, pct_displaced
from reblock.eval.kcomplexity import KComplexityEval
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, region_reblock
from reblock.render import frame_bbox, render_after, save_render


@dataclass(frozen=True)
class LensARow:
    method: str
    reached: bool           # did the method reach access-depth <= target_depth?
    reached_depth: int      # the prefix's actual max access-depth (the floor when not reached)
    road_length_m: float
    displacement: float     # Sigma disk-graze probability at the depth-D prefix
    pct_displaced: float
    propose_seconds: float  # wall-clock to reblock the method (overprovisioned), passed through


@dataclass(frozen=True)
class LensBRow:
    method: str
    budget_m: float
    external_connectivity: float
    internal_connectivity: float
    displacement: float
    pct_displaced: float


def two_lens_rows(block: Block, roads_by_method: dict[str, GeoDataFrame],
                  propose_seconds: dict[str, float], target_depth: int, budget_m: float, *,
                  corridor_m: float = 3.0) -> tuple[list[LensARow], list[LensBRow]]:
    """Pure two-lens table logic (no I/O, no rendering). For each method's full road set:
    Lens A truncates to the depth-`target_depth` prefix (`prefix_to_depth`); Lens B truncates to
    the shared `budget_m` (`truncate_to_length`) and scores external (`access_benefit`) + internal
    (`commute_ratio`) connectivity. `propose_seconds` is the caller-measured reblock time per
    method, reported verbatim (kept out of this function so it stays deterministic)."""
    radii = building_radii(block.building_points, corridor_m)
    ext_factory = access_benefit(block, None)
    lens_a: list[LensARow] = []
    lens_b: list[LensBRow] = []
    for name, roads in roads_by_method.items():
        prefix_a, reached_depth = prefix_to_depth(block, roads, target_depth)
        lens_a.append(LensARow(
            method=name, reached=reached_depth <= target_depth, reached_depth=reached_depth,
            road_length_m=float(prefix_a.geometry.length.sum()),
            displacement=displacement(block.building_points, radii, prefix_a, corridor_m),
            pct_displaced=pct_displaced(prefix_a, corridor_m, block.building_points, radii),
            propose_seconds=propose_seconds[name]))
        prefix_b = truncate_to_length(block, roads, budget_m)
        lens_b.append(LensBRow(
            method=name, budget_m=budget_m,
            external_connectivity=ext_factory(prefix_b),
            internal_connectivity=commute_ratio(block, prefix_b),
            displacement=displacement(block.building_points, radii, prefix_b, corridor_m),
            pct_displaced=pct_displaced(prefix_b, corridor_m, block.building_points, radii)))
    return lens_a, lens_b


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run_two_lens(region: list[Block], methods: dict[str, Method], target_depth: int,
                 out_dir: Path, *, corridor_m: float = 3.0
                 ) -> tuple[list[LensARow], list[LensBRow]]:
    """Reblock each method once over `region` (timed), compute both lens tables, write the two CSVs,
    and render one after-heatmap per method per lens. The region block is method-independent (same
    parcels/streets every reblock), so the first method's block scores every method and fixes the
    shared render `vmax`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    roads_by_method: dict[str, GeoDataFrame] = {}
    proposals: dict[str, Proposal] = {}
    propose_seconds: dict[str, float] = {}
    block: Block | None = None
    for name, method in methods.items():
        t0 = time.perf_counter()
        result = region_reblock(region, method, [])
        propose_seconds[name] = time.perf_counter() - t0
        block = result.block
        proposals[name] = result.proposal
        roads_by_method[name] = cast(GeoDataFrame, result.proposal.roads)
    assert block is not None
    budget = matched_budget({n: float(r.geometry.length.sum()) for n, r in roads_by_method.items()})
    lens_a, lens_b = two_lens_rows(block, roads_by_method, propose_seconds, target_depth, budget,
                                   corridor_m=corridor_m)

    kc_eval = KComplexityEval()
    vmax: int | None = None
    for name in methods:
        prefix_a, _ = prefix_to_depth(block, roads_by_method[name], target_depth)
        prefix_b = truncate_to_length(block, roads_by_method[name], budget)
        for prefix, tag in ((prefix_a, f"depth{target_depth}"), (prefix_b, "matched")):
            truncated = replace(proposals[name], roads=prefix, block_identity=None)
            kc = kc_eval.score(block, truncated)
            if vmax is None:
                vmax = int(kc.fields["access_before"].max())
            fig = render_after(block, truncated, kc.fields["access_after"], vmax=vmax, metrics=kc,
                               frame=frame_bbox(block.parcels),
                               displaced_points=_displaced_points(block, truncated))
            save_render(fig, out_dir / f"after_{name}_{tag}.jpg")
            plt.close(fig)

    _write_csv(out_dir / "lens_a_depth.csv",
               ["method", "target_depth", "reached", "reached_depth", "road_length_m",
                "displacement", "pct_displaced", "propose_seconds"],
               [[r.method, target_depth, r.reached, r.reached_depth, f"{r.road_length_m:.1f}",
                 f"{r.displacement:.1f}", f"{r.pct_displaced:.4f}", f"{r.propose_seconds:.1f}"]
                for r in lens_a])
    _write_csv(out_dir / "lens_b_matched.csv",
               ["method", "budget_m", "external_connectivity", "internal_connectivity",
                "displacement", "pct_displaced"],
               [[r.method, f"{r.budget_m:.1f}", f"{r.external_connectivity:.6g}",
                 f"{r.internal_connectivity:.6g}", f"{r.displacement:.1f}",
                 f"{r.pct_displaced:.4f}"]
                for r in lens_b])
    return lens_a, lens_b


def main() -> None:
    out_dir = Path(sys.argv[1])
    target_depth = int(sys.argv[2])
    method_names = sys.argv[3].split(",")
    overrides = ["max_blocks=1", *sys.argv[4:]]
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=overrides)
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    groups = [[str(b) for b in g] for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, 1)[0]
    methods = {n: cast(Method, instantiate(cfg.all_methods[n])) for n in method_names}
    lens_a, lens_b = run_two_lens(region, methods, target_depth, out_dir)
    for a in lens_a:
        mark = "reached" if a.reached else f"FLOOR depth {a.reached_depth}"
        print(f"[lens A d<={target_depth}] {a.method}: {mark} at {a.road_length_m:.0f} m, "
              f"{a.displacement:.0f} displaced, {a.propose_seconds:.1f} s")
    for b in lens_b:
        print(f"[lens B {b.budget_m:.0f} m] {b.method}: ext={b.external_connectivity:.3f} "
              f"int={b.internal_connectivity:.3f} {b.displacement:.0f} displaced")


if __name__ == "__main__":
    main()
