"""Is the `max_anchors` win about anchor SPREAD, or just about sampling a bigger FRACTION?

The proposed mechanism has two components tangled together, and the region result cannot separate
them. At a fixed shortlist of 512, `cap=128` scores 512 of 30,353 candidates (**1.7%**) while
uncapped scores 512 of 1,180,388 (**0.04%**). So the capped arm differs in two ways at once:

  1. **spread** -- arc-length anchors are evenly spaced by construction, while vertex anchors pile
     up wherever the parcel-boundary graph is geometrically dense;
  2. **fraction** -- it searches 40x more of its own candidate set.

If (2) is doing the work, `max_anchors` is not really the lever; the shortlist is, and the honest
recommendation would be "score more candidates", not "cap anchors". If (1) is doing the work, the
anchor family itself is better and the cap is the right knob.

The separation: hold the anchor family at uncapped and climb the shortlist. Matching the capped
arm's 1.7% would need a shortlist near 20,000, which is not affordable -- but the SHAPE of the
curve settles it without reaching that point. If permeability barely moves across a 4x-8x increase
in shortlist, extrapolating the remaining 40x to close a +0.088 gap is not credible; if it climbs
steeply, the gap is a sampling-density artifact and the whole framing changes.

Cost is dominated by exact scoring, which is linear in the shortlist (enumeration and bulk ranking
are unchanged, since the candidate set is identical across the uncapped arms). Measured on this
region, one step's 512 exact evaluations cost roughly 46 s at 16 workers, so each doubling adds
about that much per step again. Arms therefore run cheapest-first, and the JSON is rewritten after
each one -- the ladder is informative even if the expensive top rung never lands.

`MAX_ROADS` is 8 and the region is index 0, matching `region_cap_replicate.py` so the arms here sit
on the same footing as the replication rather than forming a separate baseline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from reblock.budget import building_radii, prefix_to_displacement
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.eval.access_burden import burden
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, permeability
from scripts.perf import region_pool
from scripts.perf.selectors import FirstOrder
from scripts.perf.shortlist_greedy import greedy_shortlist

REGION = 0
MAX_ROADS = 8
THREADS = 8
WORKERS = 16
BUDGETS = (0.05, 0.10, 0.15, 0.20)
# (label, max_anchors, shortlist) -- cheapest first. The cap=128 arm is the target the uncapped
# ladder is climbing toward; everything above it holds the anchor family fixed and varies only
# how much of the SAME candidate set gets scored exactly.
ARMS = (
    ("cap128-s512", 128, 512),
    ("uncapped-s512", 0, 512),
    ("uncapped-s1024", 0, 1024),
    ("uncapped-s2048", 0, 2048),
    ("uncapped-s4096", 0, 4096),
)
OUT = Path("scripts/perf/region_shortlist_confound.json")


def main() -> None:
    block = region_pool.blocks(REGION + 1)[REGION]
    n = len(block.parcels)
    half_w = DEFAULT_ROAD_WIDTH_M / 2.0
    print(f"\nregion {REGION}: {n:,} parcels, {len(block.building_points):,} buildings\n",
          flush=True)

    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    radii = building_radii(block.building_points)
    b0 = burden(parcel_access_layers(block, None, tol=STREET_TOL, adj=adj, unreached_depth=n + 1))

    out: dict[str, dict[str, object]] = {}
    for label, cap, short in ARMS:
        per_step: list[int] = []
        t0 = time.perf_counter()
        mark = [t0]

        def tick(step: int, n_cand: int, n_roads: int, acc: list[int] = per_step,
                 m: list[float] = mark, start: float = t0, tag: str = label) -> None:
            now = time.perf_counter()
            acc.append(n_cand)
            print(f"    [{tag}] step {step:>2}: {n_cand:>9,} cand  {now - m[0]:6.1f} s  "
                  f"(total {(now - start) / 60:5.1f} min)", flush=True)
            m[0] = now

        roads = greedy_shortlist(block, mode="buildable", objective="access", cost="displacement",
                                 half_width_m=half_w, workers=WORKERS, max_roads=MAX_ROADS,
                                 max_anchors=cap, selector=FirstOrder(short, threads=THREADS),
                                 on_step=tick)
        dt = time.perf_counter() - t0
        if roads is None or len(roads) == 0:
            print(f"    [{label}] no roads -- skipped", flush=True)
            continue

        at: dict[str, dict[str, float]] = {}
        for d in BUDGETS:
            pre = prefix_to_displacement(block, roads, radii, d)
            if len(pre) == 0:
                continue
            b1 = burden(parcel_access_layers(block, pre, tol=STREET_TOL, adj=adj,
                                             unreached_depth=n + 1))
            at[f"{d:.2f}"] = {"burden_red": (1.0 - b1 / b0) if b0 > 0 else 0.0,
                              "perm": float(permeability(block, pre)),
                              "road_m": float(pre.geometry.length.sum()),
                              "n_roads": float(len(pre))}
        out[label] = {"max_anchors": cap, "shortlist": short, "secs": dt, "cand": per_step,
                      "at": at, "roads_wkt": [g.wkt for g in roads.geometry]}
        frac = short / max(per_step[-1], 1) if per_step else 0.0
        print(f"    [{label}] {dt / 60:.1f} min  scored {frac:.2%} of the last step's candidates"
              + "  " + "  ".join(f"d={k} p={v['perm']:.4f}" for k, v in at.items()), flush=True)
        OUT.write_text(json.dumps(out, indent=1))

    if not out:
        print("no arms completed")
        return
    _report(out)


def _report(out: dict[str, dict[str, object]]) -> None:
    print(f"\n{'=' * 104}\nSHORTLIST CONFOUND -- region {REGION}, max_roads={MAX_ROADS}\n")
    print(f"  {'arm':<17}{'anchors':>9}{'shortlist':>11}{'cand(last)':>12}{'scored%':>9}"
          f"{'min':>7}" + "".join(f"{f'perm d={d:.2f}':>14}" for d in BUDGETS))
    for label, v in out.items():
        cand = v["cand"]
        assert isinstance(cand, list)
        last = cand[-1] if cand else 0
        short = int(v["shortlist"])  # type: ignore[arg-type]
        at = v["at"]
        assert isinstance(at, dict)
        cells = "".join(f"{at[f'{d:.2f}']['perm']:>14.4f}" if f"{d:.2f}" in at else f"{'--':>14}"
                        for d in BUDGETS)
        anch = "uncapped" if int(v["max_anchors"]) == 0 else str(v["max_anchors"])  # type: ignore[arg-type]
        print(f"  {label:<17}{anch:>9}{short:>11}{last:>12,}{short / max(last, 1):>8.2%}"
              f"{float(v['secs']) / 60:>7.1f}" + cells)  # type: ignore[arg-type]

    ladder = [(k, v) for k, v in out.items() if int(v["max_anchors"]) == 0]  # type: ignore[arg-type]
    target = out.get("cap128-s512")
    if len(ladder) >= 2 and target is not None:
        tat = target["at"]
        assert isinstance(tat, dict)
        print("\n  DOES THE LADDER CLOSE THE GAP? uncapped perm minus cap=128 perm, per budget.\n"
              "  Climbing toward 0 as shortlist rises => the win is sampling DENSITY and the\n"
              "  honest lever is the shortlist, not the cap. Flat => the win is anchor SPREAD\n"
              "  and the cap is the right knob.\n")
        print(f"    {'shortlist':>10}" + "".join(f"{f'd={d:.2f}':>11}" for d in BUDGETS))
        for _, v in ladder:
            at = v["at"]
            assert isinstance(at, dict)
            cells = "".join(
                f"{at[f'{d:.2f}']['perm'] - tat[f'{d:.2f}']['perm']:>+11.4f}"
                if f"{d:.2f}" in at and f"{d:.2f}" in tat else f"{'--':>11}" for d in BUDGETS)
            print(f"    {int(v['shortlist']):>10}" + cells)  # type: ignore[arg-type]
        print("\n  Matching cap=128's sampled fraction needs a shortlist near 20,000, which is\n"
              "  not affordable -- so read the SHAPE, not the endpoint. A gap that barely moves\n"
              "  across a 4-8x rise will not be closed by the remaining 40x.")


if __name__ == "__main__":
    main()
