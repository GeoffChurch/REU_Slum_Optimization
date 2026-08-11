"""Does the `max_anchors` region result replicate across independent regions and sizes?

`region_anchor_cap.py` measured it once: on one region block, at one displacement budget (0.10),
`cap=128` beat uncapped by +0.0884 permeability while running 8.2x faster. Two independent caps
agreeing, plus a large structural difference at equal road length, is real evidence -- but n=1
supports no interval, and a single budget cannot distinguish "better network" from "better at one
truncation point".

Three caveats closed here, one left open:

  * **n=1** -> independent regions from `region_pool`, same arms in each.
  * **one displacement budget** -> the FULL road list is persisted, so any budget becomes a
    re-analysis rather than a re-run. The predecessor saved only the 0.10 prefix, which is why this
    needed a re-run at all.
  * **cap=128 vs cap=256 unseparated** -> both run in every region, so a consistent winner would
    show as a sign that does not flip.
  * NOT closed here: whether the win is an artifact of the fixed shortlist budget interacting with
    candidate-set size. That is `region_shortlist_confound.py`, and it is the mechanism question
    rather than the robustness question.

**This harness deliberately does NOT evaluate at absolute displacement budgets.** Region networks
displace only ~0.005-0.02, so block-scale budgets like 0.10 are unreachable, and
`prefix_to_displacement` silently returns *all* roads when a budget cannot be met -- which is how
the original headline came to compare arms that had spent 68% different displacement while claiming
to be matched. Reachability is a property of the arms, so matching is computed per region *after*
every arm exists: see `region_cap_matched.py`. What is recorded here is each arm's full road list
and the displacement it actually achieves.

`MAX_ROADS` is 8 rather than the headline's 15 because uncapped costs ~31 min at 8 and ~80 at 15,
and replication needs several regions. Region 0 is re-run at 8 here despite already having 15-road
numbers, so the replication set is internally consistent and the road-count difference becomes
visible rather than confounded.

Arms run cheapest-first (128, 256, uncapped) and the JSON is rewritten after every arm: uncapped is
~80% of the wall clock, four background runs on this machine have been killed for unknown reasons,
and a kill should cost the expensive arm rather than the whole matrix.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from reblock.budget import building_radii, displacement
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.eval.access_burden import burden
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, permeability
from scripts.perf import region_pool
from scripts.perf.selectors import FirstOrder
from scripts.perf.shortlist_greedy import greedy_shortlist

N_REGIONS = 6
SHORTLIST = 512
MAX_ROADS = 8
THREADS = 8
WORKERS = 16
CAPS = (128, 256, 0)                  # cheapest first; 0 == uncapped == shipped default
OUT = Path("scripts/perf/region_cap_replicate.json")


def main() -> None:
    pool = region_pool.blocks(N_REGIONS)
    half_w = DEFAULT_ROAD_WIDTH_M / 2.0
    out: dict[str, dict[str, object]] = {}

    # Ascending size: uncapped cost grows ~quadratically in parcels (3.4k ~3 min, 12k ~35), so
    # cheapest-first buys the most data points per minute survived. It also builds the SIZE
    # GRADIENT from the cheap end, which is the real prize here -- the mechanism says the cap wins
    # because vertex anchors dominate and cluster at scale, so the effect should GROW with region
    # size and vanish toward block scale, where it already measured neutral-to-negative. A
    # monotone trend across 3.4k -> 12k is dose-response evidence replication alone cannot give.
    order = sorted(range(len(pool)), key=lambda i: len(pool[i].parcels))
    for ri in order:
        block = pool[ri]
        n = len(block.parcels)
        print(f"\n=== region {ri}: {n:,} parcels, {len(block.building_points):,} buildings ===",
              flush=True)
        adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
        radii = building_radii(block.building_points)
        b0 = burden(parcel_access_layers(block, None, tol=STREET_TOL, adj=adj,
                                         unreached_depth=n + 1))
        arms: dict[str, dict[str, object]] = {}

        for cap in CAPS:
            label = "uncapped" if cap == 0 else str(cap)
            per_step: list[int] = []
            t0 = time.perf_counter()
            mark = [t0]

            def tick(step: int, n_cand: int, n_roads: int, acc: list[int] = per_step,
                     m: list[float] = mark, start: float = t0, tag: str = f"r{ri} {label}") -> None:
                now = time.perf_counter()
                acc.append(n_cand)
                print(f"    [{tag}] step {step:>2}: {n_cand:>9,} cand  "
                      f"{now - m[0]:6.1f} s  (total {(now - start) / 60:5.1f} min)", flush=True)
                m[0] = now

            roads = greedy_shortlist(block, mode="buildable", objective="access",
                                     cost="displacement", half_width_m=half_w, workers=WORKERS,
                                     max_roads=MAX_ROADS, max_anchors=cap,
                                     selector=FirstOrder(SHORTLIST, threads=THREADS),
                                     on_step=tick)
            dt = time.perf_counter() - t0
            if roads is None or len(roads) == 0:
                print(f"    [r{ri} {label}] no roads -- skipped", flush=True)
                continue

            # Whole-network figures only. Matching happens in region_cap_matched.py, which can see
            # every arm's reachable displacement and pick a budget that actually binds; an absolute
            # budget chosen here would silently degrade to "all roads" and fake a matched result.
            nb = len(block.building_points)
            reach = displacement(block.building_points, radii, roads) / nb if nb else 0.0
            b1 = burden(parcel_access_layers(block, roads, tol=STREET_TOL, adj=adj,
                                             unreached_depth=n + 1))
            at = {"all": {"burden_red": (1.0 - b1 / b0) if b0 > 0 else 0.0,
                          "perm": float(permeability(block, roads)),
                          "road_m": float(roads.geometry.length.sum()),
                          "n_roads": float(len(roads)), "displaced_frac": float(reach)}}
            # roads_wkt is the FULL list, not a prefix, so any future budget question is a
            # re-analysis rather than a re-run -- the flaw that forced this harness to exist.
            arms[label] = {"secs": dt, "cand": per_step, "at": at,
                           "roads_wkt": [g.wkt for g in roads.geometry]}
            a = at["all"]
            print(f"    [r{ri} {label}] {dt / 60:.1f} min  whole network: "
                  f"b={a['burden_red']:.4f} p={a['perm']:.4f} "
                  f"{a['n_roads']:.0f} roads {a['road_m']:,.0f} m "
                  f"displaced {a['displaced_frac']:.4f}", flush=True)
            out[str(ri)] = {"parcels": n, "arms": arms}
            OUT.write_text(json.dumps(out, indent=1))

    if not out:
        print("no arms completed")
        return
    _report(out)


def _report(out: dict[str, dict[str, object]]) -> None:
    """Speed, enumeration growth and displacement reach -- everything that needs no matching.

    Quality is deliberately absent. Comparing burden/perm across arms requires equal displacement,
    the arms reach different amounts, and a budget picked here cannot know what is reachable until
    every arm has run. That comparison lives in `region_cap_matched.py`.
    """
    ids = sorted(out, key=lambda r: int(out[r]["parcels"]))  # type: ignore[arg-type]
    print(f"\n{'=' * 100}\nREGION CAP REPLICATION -- {len(out)} regions, max_roads={MAX_ROADS}, "
          f"shortlist={SHORTLIST}\n")

    print("  SPEED and ENUMERATION (ascending region size)")
    print(f"    {'region':<8}{'parcels':>9}{'arm':>10}{'min':>8}{'speedup':>9}"
          f"{'cand step1':>12}{'cand last':>11}{'growth':>8}{'displaced':>11}")
    for ri in ids:
        arms = out[ri]["arms"]
        assert isinstance(arms, dict)
        if "uncapped" not in arms:
            continue
        base = float(arms["uncapped"]["secs"])
        for lb in ("uncapped", "128", "256"):
            if lb not in arms:
                continue
            v = arms[lb]
            cand = v["cand"]
            assert isinstance(cand, list)
            f, ln = (cand[0], cand[-1]) if cand else (0, 0)
            a = v["at"]["all"]
            print(f"    {ri:<8}{int(out[ri]['parcels']):>9,}{lb:>10}"  # type: ignore[arg-type]
                  f"{float(v['secs']) / 60:>8.1f}{base / float(v['secs']):>8.1f}x"
                  f"{f:>12,}{ln:>11,}{ln / max(f, 1):>7.2f}x{a['displaced_frac']:>11.4f}")

    print("\n  The displacement column is why quality is not compared here: the arms spend\n"
          "  different amounts of it, and displacement buys both burden and permeability.\n"
          "  Run `python -m scripts.perf.region_cap_matched` for the matched comparison.")
