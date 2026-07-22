"""Two-lens method comparison for the multiblock example.

Reblocks each method once over the region (timed), then reports it under two budgets:

  Lens A -- fixed OUTCOME (external-connectivity target E): the drainage-ordered road prefix that
    first brings external connectivity (`budget.access_benefit`) to >= E
    (`budget.prefix_to_external_connectivity`); reports the road length, displacement and
    wall-clock propose time it took. A fixed input that never reaches E (osm_footpaths) is
    reported unreached with its floor connectivity.

  Lens B -- fixed COST (matched road budget): every method truncated to the sparsest method's
    total added road length (`budget.matched_budget` + `truncate_to_length`); reports benefit on
    both axes (external + internal connectivity) + displacement.

Both lenses render one after-heatmap per method (Lens A at the connectivity-E prefix, Lens B at the
matched budget), re-scoring access-depth on each truncated road set via `KComplexityEval` (the same
eval `region_reblock`/`pipeline.run` invoke, called directly on a Proposal wrapping the truncated
roads with `block_identity=None` so the derive memo never hands back the untruncated depth).

Run (module form -- mirrors scripts/fetch_desire_lines_snapshot.py's Hydra bootstrapping):
  pixi run python -m scripts.compare_budgets <out_dir> <target_ext> <m1,m2,...> \
       <hydra override>...

  e.g. examples/multiblock 0.70 clearance,greedy_arterial_buildable,osm_footpaths \
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

from reblock.animate import reblock_gif
from reblock.budget import (
    _commute_membership,
    access_benefit,
    building_radii,
    commute_ratio,
    commute_ratio_benefit,
    cost_benefit_curve,
    displacement,
    displacement_curve,
    matched_budget,
    prefix_to_external_connectivity,
    truncate_to_length,
)
from reblock.compare import MethodCurve
from reblock.contracts import Block, Method, Proposal, Screen, Source
from reblock.emit import (
    _displaced_points,
    compare_report,
    depth_vs_road_report,
    pct_displaced,
    pct_paved,
)
from reblock.eval.kcomplexity import KComplexityEval
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, region_reblock
from reblock.render import frame_bbox, render_after, save_render, short_label


@dataclass(frozen=True)
class LensARow:
    method: str
    reached: bool           # did the method reach external connectivity >= target_ext?
    reached_ext: float      # the prefix's actual external connectivity (the floor when not reached)
    road_length_m: float
    displacement: float     # Sigma disk-graze probability at the connectivity-E prefix
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
                  propose_seconds: dict[str, float], target_ext: float, budget_m: float, *,
                  corridor_m: float = 3.0) -> tuple[list[LensARow], list[LensBRow]]:
    """Pure two-lens table logic (no I/O, no rendering). For each method's full road set:
    Lens A truncates to the external-connectivity-`target_ext` prefix
    (`prefix_to_external_connectivity`); Lens B truncates to the shared `budget_m`
    (`truncate_to_length`) and scores external (`access_benefit`) + internal (`commute_ratio`)
    connectivity -- the internal scalar is frozen to the method's full network (membership from
    `roads`), matching the (now frozen) internal curve. `propose_seconds` is the caller-measured
    reblock time per method, reported verbatim (kept out of this function so it stays
    deterministic)."""
    radii = building_radii(block.building_points, corridor_m)
    ext_factory = access_benefit(block, None)
    lens_a: list[LensARow] = []
    lens_b: list[LensBRow] = []
    for name, roads in roads_by_method.items():
        prefix_a, reached_ext = prefix_to_external_connectivity(block, roads, target_ext)
        lens_a.append(LensARow(
            method=name, reached=reached_ext >= target_ext, reached_ext=reached_ext,
            road_length_m=float(prefix_a.geometry.length.sum()),
            displacement=displacement(block.building_points, radii, prefix_a, corridor_m),
            pct_displaced=pct_displaced(prefix_a, corridor_m, block.building_points, radii),
            propose_seconds=propose_seconds[name]))
        prefix_b = truncate_to_length(block, roads, budget_m)
        internal_membership = _commute_membership(block, roads)  # freeze to method's FULL network
        lens_b.append(LensBRow(
            method=name, budget_m=budget_m,
            external_connectivity=ext_factory(prefix_b),
            internal_connectivity=commute_ratio(block, prefix_b, membership=internal_membership),
            displacement=displacement(block.building_points, radii, prefix_b, corridor_m),
            pct_displaced=pct_displaced(prefix_b, corridor_m, block.building_points, radii)))
    return lens_a, lens_b


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run_two_lens(region: list[Block], methods: dict[str, Method], target_ext: float,
                 out_dir: Path, *, corridor_m: float = 3.0, label: str | None = None,
                 extend: dict[str, Method] | None = None
                 ) -> tuple[list[LensARow], list[LensBRow]]:
    """Reblock each method once over `region` (timed), compute both lens tables, write the two CSVs,
    and render one after-heatmap per method per lens. The region block is method-independent (same
    parcels/streets every reblock), so any method's block scores every method and fixes the
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
    lens_a, lens_b = two_lens_rows(block, roads_by_method, propose_seconds, target_ext, budget,
                                   corridor_m=corridor_m)

    kc_eval = KComplexityEval()
    vmax: int | None = None
    for name in methods:
        prefix_a, _ = prefix_to_external_connectivity(block, roads_by_method[name], target_ext)
        prefix_b = truncate_to_length(block, roads_by_method[name], budget)
        ext_tag = f"ext{int(round(target_ext * 100))}"
        for prefix, tag in ((prefix_a, ext_tag), (prefix_b, "matched")):
            truncated = replace(proposals[name], roads=prefix, block_identity=None)
            kc = kc_eval.score(block, truncated)
            if vmax is None:
                vmax = int(kc.fields["access_before"].max())
            fig = render_after(block, truncated, kc.fields["access_after"], vmax=vmax, metrics=kc,
                               frame=frame_bbox(block.parcels),
                               displaced_points=_displaced_points(block, truncated))
            save_render(fig, out_dir / f"after_{name}_{tag}.jpg")
            plt.close(fig)

    # Per-method reblock GIF: the method's full road set added in drainage order, the region
    # draining on the same access-depth scale. Frames render across a fork pool (reblock.animate).
    assert vmax is not None                        # set by the loop above (methods is non-empty)
    frame = frame_bbox(block.parcels)
    for name, roads in roads_by_method.items():
        reblock_gif(block, roads, out_dir / f"reblock_{name}.gif", vmax=vmax, frame=frame)

    _write_csv(out_dir / "lens_a_external.csv",
               ["method", "target_ext", "reached", "reached_ext", "road_length_m",
                "displacement", "pct_displaced", "propose_seconds"],
               [[r.method, f"{target_ext:.2f}", r.reached, f"{r.reached_ext:.4f}",
                 f"{r.road_length_m:.1f}", f"{r.displacement:.1f}", f"{r.pct_displaced:.4f}",
                 f"{r.propose_seconds:.1f}"]
                for r in lens_a])
    _write_csv(out_dir / "lens_b_matched.csv",
               ["method", "budget_m", "external_connectivity", "internal_connectivity",
                "displacement", "pct_displaced"],
               [[r.method, f"{r.budget_m:.1f}", f"{r.external_connectivity:.6g}",
                 f"{r.internal_connectivity:.6g}", f"{r.displacement:.1f}",
                 f"{r.pct_displaced:.4f}"]
                for r in lens_b])

    # Frontier curves: each method's full (added-road-length, benefit) trade-off, built from the
    # SAME reblock as the two lenses above (`roads_by_method`) -- no second propose. The
    # fixed-connectivity and matched-budget lens rows are just points on these curves.
    # `compare_report` writes curve_{external,internal}_connectivity_<label>.png +
    # displacement_<label>.png + frontier CSVs.
    radii = building_radii(block.building_points, corridor_m)
    block_area = float(block.parcels.geometry.union_all().area)
    # Label the curve files by the caller-supplied id (the region's seed/top-scoring block, so they
    # correlate with the README's §1 block), falling back to the first member for standalone runs.
    curve_label = short_label(label if label is not None else str(region[0].block_id))
    curves: list[MethodCurve] = []
    for name, roads in roads_by_method.items():
        pp = pct_paved(roads, corridor_m, block_area)
        pd_ = pct_displaced(roads, corridor_m, block.building_points, radii)
        ext = cost_benefit_curve(block, roads, benefit_fn=access_benefit)
        internal = cost_benefit_curve(block, roads, benefit_fn=commute_ratio_benefit)
        disp = displacement_curve(block, roads, radii, corridor_m=corridor_m)
        curves.append(MethodCurve(name, curve_label, "external_connectivity", ext, pp, pd_))
        curves.append(MethodCurve(name, curve_label, "internal_connectivity", internal, pp, pd_))
        curves.append(MethodCurve(name, curve_label, "displacement", disp, pp, pd_))
    compare_report(curves, out_dir, method_order=list(methods))
    # Access-depth vs road. `extend` supplies over-provisioned re-runs of the method(s) to continue;
    # each is truncated in drainage order to L_max = the longest CONVERGED road (max over the normal
    # proposals), so it keeps clearing past its depth target out to L_max. A method NOT in `extend`
    # (osm_footpaths, which is fixed; arterial, sparse by design) stops at its own converged length.
    curve_roads = dict(roads_by_method)
    if extend:
        converged = {n: float(r.geometry.length.sum()) for n, r in roads_by_method.items()}
        l_max = max(converged.values())
        for name, xmethod in extend.items():
            if converged.get(name, 0.0) >= l_max - 1e-6:
                continue      # already the longest -> its own road is the curve road; no re-run
            xroads = cast(GeoDataFrame, region_reblock(region, xmethod, []).proposal.roads)
            curve_roads[name] = truncate_to_length(block, xroads, l_max)
    depth_vs_road_report(block, curve_roads, out_dir, method_order=list(methods), label=curve_label)
    return lens_a, lens_b


def main() -> None:
    out_dir = Path(sys.argv[1])
    target_ext = float(sys.argv[2])
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
    lens_a, lens_b = run_two_lens(region, methods, target_ext, out_dir)
    for a in lens_a:
        mark = f"reached ext {a.reached_ext:.2f}" if a.reached else f"FLOOR ext {a.reached_ext:.2f}"
        print(f"[lens A ext>={target_ext:.2f}] {a.method}: {mark} at {a.road_length_m:.0f} m, "
              f"{a.displacement:.0f} displaced, {a.propose_seconds:.1f} s")
    for b in lens_b:
        print(f"[lens B {b.budget_m:.0f} m] {b.method}: ext={b.external_connectivity:.3f} "
              f"int={b.internal_connectivity:.3f} {b.displacement:.0f} displaced")


if __name__ == "__main__":
    main()
