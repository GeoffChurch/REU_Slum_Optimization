"""Permeability two-lens method comparison.

Reblocks each method once over the region (natural config -- no over-provisioning), then reports
it three ways, all from that SAME reblock (no second propose):

  Frontier -- the permeability-vs-displacement curve (`permeability_curve` + `displacement_curve`,
    index-aligned over the same drainage-ordered `_sweep`) for every method, overlaid via
    `emit.compare_report`.

  Lens A -- matched displacement (a universal displacement fraction `D`): the drainage-ordered road
    prefix that first brings the displacement fraction to >= `D` (`budget.prefix_to_displacement`);
    reports the permeability reached at that equal home-cost. The METRIC is monotone (a longer
    prefix never displaces fewer homes), but a method's own network can still exhaust itself short
    of `D` -- many methods converge well below a demanding `D` (see the calibration probe) -- so
    `OutcomeRow.at_budget`/the after-image title report that honestly ("converged at X% (< D%
    budget)") rather than implying every method landed AT `D`.

  Lens B -- matched permeability (a universal permeability level `P*`): the drainage-ordered road
    prefix that first brings permeability to >= `P*` (`budget.prefix_to_permeability`); reports the
    displacement spent to reach it. A method whose full network never reaches `P*` (e.g.
    osm_footpaths, a fixed input) is reported unreached, at its floor permeability's best-effort
    (all roads) prefix.

Both lenses render one after-heatmap per method, in BOTH colorings (access-depth via
`KComplexityEval`'s `access_after`, and the permeability potential via
`reblock.permeability.parcel_potentials`) -- re-scoring against the truncated road set via a
`Proposal` with `block_identity=None` so the derive memo never hands back the untruncated depth.
One before-heatmap per region, also in both colorings. One reblock-drainage GIF per method
(unchanged, full network).

`load_permeability_config` reads `conf/permeability.yaml` (metric params + the two calibrated
thresholds `matched_displacement`/`matched_permeability`) -- the single source both example
generators and this module's own CLI load from, so a re-calibration only ever touches the yaml.

Run (module form -- mirrors scripts/fetch_desire_lines_snapshot.py's Hydra bootstrapping):
  pixi run python -m scripts.compare_budgets <out_dir> <m1,m2,...> <hydra override>...

  e.g. examples/multiblock clearance,greedy_arterial_buildable,osm_footpaths \
       data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
       block_ids=[[ZAF.9.3.1_1_5810]] all_methods.clearance.max_roads=3000 \
       all_methods.clearance.depth_target=3 \
       all_methods.greedy_arterial_buildable.engine.policy._target_=reblock.methods.arterial.Fixed \
       +all_methods.greedy_arterial_buildable.max_anchors=64 \
       desire_source.snapshot=examples/multiblock/desire_lines_5810.geojson
"""
from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from reblock.animate import reblock_gif
from reblock.budget import (
    building_radii,
    displacement,
    displacement_curve,
    prefix_to_displacement,
    prefix_to_permeability,
)
from reblock.compare import MethodCurve
from reblock.contracts import Block, Method, Proposal, Screen, Source
from reblock.derivations import propose
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.emit import _displaced_points, compare_report
from reblock.eval.access_burden import burden
from reblock.eval.kcomplexity import KComplexityEval
from reblock.permeability import (
    PermeabilityParams,
    egress_power,
    parcel_potentials,
    permeability,
    permeability_curve,
)
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, region_reblock
from reblock.render import frame_bbox, render_after, render_before, save_render, short_label

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutcomeRow:
    method: str
    disp_road_m: float        # Lens A: road length at the matched-displacement prefix
    disp_permeability: float  # Lens A: permeability reached at that prefix
    at_budget: bool           # Lens A: True iff the method's prefix actually reached D (not just
                               # converged below it and shown at its own terminal -- displacement
                               # is monotone, but a method can still exhaust its OWN roads short of
                               # D; see `prefix_to_displacement`'s "unreachable: best effort" path)
    perm_road_m: float        # Lens B: road length at the matched-permeability prefix
    perm_displacement: float  # Lens B: displacement fraction spent to reach it
    reached: bool             # Lens B: whether matched_permeability was actually reached
    # ACCESS, the second reported axis, at each lens's prefix: 1 - burden(roads)/burden(no roads),
    # burden being sum (depth-1)^2 / n (reblock.eval.access_burden). Deliberately the same shape as
    # permeability so the two read alike, and reported at the SAME prefixes so a method's two
    # numbers are directly comparable. The axes agree at rho +0.810 over 10 blocks x 8 methods, but
    # not everywhere -- euclidean_grid is mid-pack on permeability and last on access -- which is
    # the reason both ship rather than one.
    disp_burden_reduction: float   # Lens A
    perm_burden_reduction: float   # Lens B


def load_permeability_config(config_dir: Path = Path("conf")
                             ) -> tuple[PermeabilityParams, float, float]:
    """`conf/permeability.yaml`'s metric params + the two calibrated lens thresholds
    (`matched_displacement` D for Lens A, `matched_permeability` P* for Lens B)."""
    raw = cast(DictConfig, OmegaConf.load(config_dir / "permeability.yaml"))
    params = PermeabilityParams(g_walk=float(raw.g_walk),
                                g_road_per_m=float(raw.g_road_per_m),
                                g_street=float(raw.g_street),
                                road_margin_m=float(raw.road_margin_m),
                                radius_frac=float(raw.radius_frac))
    return params, float(raw.matched_displacement), float(raw.matched_permeability)


def _reblock_once(region: list[Block], method: Method) -> tuple[Block, Proposal]:
    """One propose for `method` over `region`. A singleton region takes the exact pre-region
    single-block path (`propose` directly on `region[0]`) rather than `region_reblock`:
    `region_reblock`/`region_block` unions `region[0].streets` into ONE row (a single (Multi)
    LineString), which a method that filters `streets.geometry` by `isinstance(..., LineString)`
    (e.g. `TopologyMethod`, needed single-block-only by the `method_comparison`
    example variant) would then see as empty -- mirrors `reblock.compare.compare`'s
    identical singleton branch."""
    if len(region) == 1:
        block = region[0]
        return block, propose(method, block)
    result = region_reblock(region, method, [])
    return result.block, result.proposal


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run_permeability_lenses(region: list[Block], methods: dict[str, Method], out_dir: Path, *,
                            matched_displacement: float, matched_permeability: float,
                            params: PermeabilityParams,
                            label: str | None = None) -> list[OutcomeRow]:
    """Reblock each method once over `region` (natural config), compute the permeability +
    displacement frontier, both lenses' outcome tables, and every render (before, both colorings;
    per-method-per-lens after, both colorings; per-method GIF) -- all from that single reblock. The
    region block is method-independent (same parcels/streets every reblock), so any method's block
    scores every method and fixes the shared render frame/vmax."""
    out_dir.mkdir(parents=True, exist_ok=True)
    roads_by_method: dict[str, GeoDataFrame] = {}
    proposals: dict[str, Proposal] = {}
    block: Block | None = None
    for name, method in methods.items():
        t0 = time.perf_counter()
        blk, proposal = _reblock_once(region, method)
        block = blk
        proposals[name] = proposal
        roads = cast(GeoDataFrame, proposal.roads)
        roads_by_method[name] = roads
        log.info("reblocked %s: %d segments, %.0f m (%.1fs)", name, len(roads),
                 float(roads.geometry.length.sum()), time.perf_counter() - t0)
    assert block is not None

    radii = building_radii(block.building_points)
    n_buildings = len(block.building_points)

    def _disp_frac(prefix: GeoDataFrame) -> float:
        if n_buildings == 0:
            return 0.0
        return displacement(block.building_points, radii, prefix) / n_buildings

    # Frontier: permeability + displacement curves per method, from the SAME reblock above -- no
    # second propose. `compare_report` writes frontier_permeability.csv + frontier_<label>.png.
    curve_label = short_label(label if label is not None else str(region[0].block_id))
    lens_t0 = time.perf_counter()
    curves: list[MethodCurve] = []
    for name, roads in roads_by_method.items():
        curves.append(MethodCurve(name, curve_label, "permeability",
                                  permeability_curve(block, roads, params)))
        curves.append(MethodCurve(name, curve_label, "displacement",
                                  displacement_curve(block, roads, radii)))
    compare_report(curves, out_dir, method_order=list(methods),
                   matched_displacement=matched_displacement,
                   matched_permeability=matched_permeability)

    # Lens prefixes -- either lens can fall short of its target with a fixed/sparse method's own
    # network (Lens A: `at_budget=False` below, prefix_a is that method's full network shown at its
    # own terminal; Lens B: `reached=False`, prefix_b is likewise its full network).
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    p0, _ = egress_power(block, None, params, adj=adj)
    prefix_a: dict[str, GeoDataFrame] = {}
    prefix_b: dict[str, GeoDataFrame] = {}
    reached_b: dict[str, bool] = {}
    for name, roads in roads_by_method.items():
        prefix_a[name] = prefix_to_displacement(block, roads, radii, matched_displacement)
        pb, reached = prefix_to_permeability(block, roads, matched_permeability, params)
        prefix_b[name] = pb
        reached_b[name] = reached
    log.info("lens/frontier curves computed for %d methods (%.1fs)", len(methods),
             time.perf_counter() - lens_t0)

    # Before images (both colorings), once per region, on a frame + vmax shared with every after.
    kc_eval = KComplexityEval()
    frame = frame_bbox(block.parcels)
    no_roads = Proposal(block_id=block.block_id, crs=block.crs, roads=None)
    kc0 = kc_eval.score(block, no_roads)
    depth_vmax = int(kc0.fields["access_before"].max())
    fig = render_before(block, kc0.fields["access_before"], vmax=depth_vmax, field="depth",
                        frame=frame)
    save_render(fig, out_dir / "before_depth.png")
    plt.close(fig)

    potentials0 = parcel_potentials(block, None, params)
    perm_vmax = float(potentials0.max()) if len(potentials0) else 0.0
    fig = render_before(block, potentials0, vmax=perm_vmax, field="perm", frame=frame)
    save_render(fig, out_dir / "before_perm.png")
    plt.close(fig)

    # After images per method per lens, both colorings; the two outcome tables' rows.
    rows: list[OutcomeRow] = []
    disp_csv_rows: list[list[object]] = []
    perm_csv_rows: list[list[object]] = []
    # access baseline, road-independent, computed once for the region like p0 is
    burden0 = burden(parcel_access_layers(block, None, tol=STREET_TOL, adj=adj,
                                          unreached_depth=len(block.parcels) + 1))

    def _burden_red(prefix: GeoDataFrame) -> float:
        if burden0 <= 0.0:
            return 0.0
        after = burden(parcel_access_layers(block, prefix, tol=STREET_TOL, adj=adj,
                                            unreached_depth=len(block.parcels) + 1))
        return 1.0 - after / burden0

    for name in methods:
        pa, pb, reached = prefix_a[name], prefix_b[name], reached_b[name]
        disp_frac_a = _disp_frac(pa)
        perm_at_a = permeability(block, pa, params, p0=p0, adj=adj)
        disp_frac_b = _disp_frac(pb)
        perm_at_b = permeability(block, pb, params, p0=p0, adj=adj)
        # Displacement is monotone, but a method can still exhaust its OWN roads short of D (many
        # methods converge well below matched_displacement -- see the calibration probe): a method
        # is only genuinely "at the budget" if its Lens-A prefix's actual displacement reached D,
        # not merely shown at its own terminal. Framed POSITIVELY when short of it ("converged"),
        # never "unreached"/"failed" -- that framing is Lens B's, for a permeability standard a
        # method genuinely never clears.
        at_budget = disp_frac_a >= matched_displacement - 1e-9
        disp_title = (None if at_budget else
                     f"converged at {disp_frac_a * 100:.1f}% "
                     f"(< {matched_displacement * 100:.0f}% budget)")
        perm_title = None if reached else "unreached"

        for prefix, tag, title_override in ((pa, "disp", disp_title), (pb, "perm", perm_title)):
            truncated = replace(proposals[name], roads=prefix, block_identity=None)
            kc = kc_eval.score(block, truncated)
            fig = render_after(block, truncated, kc.fields["access_after"], vmax=depth_vmax,
                               metrics=kc, field="depth", frame=frame,
                               displaced_points=_displaced_points(block, truncated))
            if title_override is not None:
                fig.axes[0].set_title(title_override, fontsize=16)
            save_render(fig, out_dir / f"after_{name}_{tag}_depth.png")
            plt.close(fig)

            potentials = parcel_potentials(block, prefix, params)
            fig = render_after(block, truncated, potentials, vmax=perm_vmax, field="perm",
                               frame=frame, displaced_points=_displaced_points(block, truncated))
            if title_override is not None:
                fig.axes[0].set_title(title_override, fontsize=16)
            save_render(fig, out_dir / f"after_{name}_{tag}_perm.png")
            plt.close(fig)

        burden_red_a, burden_red_b = _burden_red(pa), _burden_red(pb)
        rows.append(OutcomeRow(
            method=name, disp_road_m=float(pa.geometry.length.sum()), disp_permeability=perm_at_a,
            at_budget=at_budget, perm_road_m=float(pb.geometry.length.sum()),
            perm_displacement=disp_frac_b, reached=reached,
            disp_burden_reduction=burden_red_a, perm_burden_reduction=burden_red_b))
        disp_csv_rows.append([name, f"{rows[-1].disp_road_m:.1f}", f"{disp_frac_a:.4f}",
                              f"{perm_at_a:.6g}", f"{burden_red_a:.6g}", at_budget])
        perm_csv_rows.append([name, f"{rows[-1].perm_road_m:.1f}", f"{disp_frac_b:.4f}",
                              f"{perm_at_b:.6g}", f"{burden_red_b:.6g}", reached])

    # Per-method reblock GIF over the FULL road set (unchanged) -- depth coloring only (the only
    # mode `reblock_gif`/`animate._frame_png` render).
    for name, roads in roads_by_method.items():
        reblock_gif(block, roads, out_dir / f"reblock_{name}.gif", vmax=depth_vmax, frame=frame)

    _write_csv(out_dir / "lens_displacement.csv",
              ["method", "road_m", "displacement", "permeability", "access_burden_reduction",
               "at_budget"], disp_csv_rows)
    _write_csv(out_dir / "lens_permeability.csv",
              ["method", "road_m", "displacement", "permeability", "access_burden_reduction",
               "reached"], perm_csv_rows)
    return rows


def main() -> None:
    out_dir = Path(sys.argv[1])
    method_names = sys.argv[2].split(",")
    overrides = ["max_blocks=1", *sys.argv[3:]]
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=overrides)
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    groups = [[str(b) for b in g] for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, 1)[0]
    methods = {n: cast(Method, instantiate(cfg.all_methods[n])) for n in method_names}
    params, matched_displacement, matched_permeability = load_permeability_config()
    rows = run_permeability_lenses(region, methods, out_dir,
                                   matched_displacement=matched_displacement,
                                   matched_permeability=matched_permeability, params=params)
    for r in rows:
        print(f"[lens A D={matched_displacement:.2f}] {r.method}: "
              f"permeability={r.disp_permeability:.3f} "
              f"access={r.disp_burden_reduction:+.3f} at {r.disp_road_m:.0f} m")
        mark = f"reached P*={matched_permeability:.2f}" if r.reached else "UNREACHED"
        print(f"[lens B P*={matched_permeability:.2f}] {r.method}: {mark}, "
              f"{r.perm_displacement * 100:.1f}% displaced, "
              f"access={r.perm_burden_reduction:+.3f} at {r.perm_road_m:.0f} m")


if __name__ == "__main__":
    main()
