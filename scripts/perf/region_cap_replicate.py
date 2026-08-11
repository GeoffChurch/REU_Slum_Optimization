"""Does the `max_anchors` region win replicate across independent regions and displacement budgets?

`region_anchor_cap.py` measured it once: on one region block, at one displacement budget (0.10),
`cap=128` beat uncapped by +0.0884 permeability while running 8.2x faster. Two independent caps
agreeing, plus a large structural difference at equal road length, is real evidence -- but n=1
supports no interval, and a single budget cannot distinguish "better network" from "better at one
truncation point".

Three caveats closed here, one left open:

  * **n=1** -> independent regions from `region_pool`, same arms in each.
  * **one displacement budget** -> the FULL road list is persisted and evaluated at four budgets.
    The previous harness saved only the 0.10 prefix, which is why this needs a re-run rather than a
    re-analysis; saving the full list means no future budget question costs compute again.
  * **cap=128 vs cap=256 unseparated** -> both run in every region, so a consistent winner would
    show as a sign that does not flip.
  * NOT closed here: whether the win is an artifact of the fixed shortlist budget interacting with
    candidate-set size. That is `region_shortlist_confound.py`, and it is the mechanism question
    rather than the robustness question.

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

from reblock.budget import building_radii, prefix_to_displacement
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
BUDGETS = (0.05, 0.10, 0.15, 0.20)
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
            # roads_wkt is the FULL list, not a prefix, so any future budget question is a
            # re-analysis rather than a re-run -- the flaw that forced this harness to exist.
            arms[label] = {"secs": dt, "cand": per_step, "at": at,
                           "roads_wkt": [g.wkt for g in roads.geometry]}
            shown = "  ".join(f"d={k} b={v['burden_red']:.4f} p={v['perm']:.4f}"
                              for k, v in at.items())
            print(f"    [r{ri} {label}] {dt / 60:.1f} min  {shown}", flush=True)
            out[str(ri)] = {"parcels": n, "arms": arms}
            OUT.write_text(json.dumps(out, indent=1))

    if not out:
        print("no arms completed")
        return
    _report(out)


def _delta(out: dict[str, dict[str, object]], ri: str, lb: str, key: str
           ) -> tuple[float, float] | None:
    """(burden, perm) of arm `lb` minus uncapped, in region `ri` at budget `key`."""
    arms = out[ri]["arms"]
    assert isinstance(arms, dict)
    if not all(x in arms and key in arms[x]["at"] for x in ("uncapped", lb)):
        return None
    ref, got = arms["uncapped"]["at"][key], arms[lb]["at"][key]
    return got["burden_red"] - ref["burden_red"], got["perm"] - ref["perm"]


def _report(out: dict[str, dict[str, object]]) -> None:
    ids = sorted(out, key=lambda r: int(out[r]["parcels"]))  # type: ignore[arg-type]
    print(f"\n{'=' * 100}\nREGION CAP REPLICATION -- {len(out)} regions, max_roads={MAX_ROADS}, "
          f"shortlist={SHORTLIST}\n")

    for d in BUDGETS:
        key = f"{d:.2f}"
        rows = [(ri, {lb: _delta(out, ri, lb, key) for lb in ("128", "256")}) for ri in ids]
        rows = [(ri, dd) for ri, dd in rows if all(v is not None for v in dd.values())]
        if not rows:
            continue
        print(f"  displacement {key} -- delta vs uncapped, per region (ascending size)")
        print(f"    {'region':<8}{'parcels':>9}{'128 burden':>13}{'128 perm':>11}"
              f"{'256 burden':>13}{'256 perm':>11}")
        for ri, dd in rows:
            print(f"    {ri:<8}{int(out[ri]['parcels']):>9,}"  # type: ignore[arg-type]
                  f"{dd['128'][0]:>+13.4f}{dd['128'][1]:>+11.4f}"  # type: ignore[index]
                  f"{dd['256'][0]:>+13.4f}{dd['256'][1]:>+11.4f}")  # type: ignore[index]
        for lb in ("128", "256"):
            pw = sum(1 for _, dd in rows if dd[lb][1] > 0)   # type: ignore[index]
            bw = sum(1 for _, dd in rows if dd[lb][0] > 0)   # type: ignore[index]
            print(f"    cap={lb}: perm improves in {pw}/{len(rows)} regions, "
                  f"burden in {bw}/{len(rows)}")
        print()

    print("  SIZE GRADIENT -- perm delta of cap=128 vs uncapped, against region size.\n"
          "  The mechanism (vertex anchors dominate and cluster as the network grows) predicts\n"
          "  the gain RISES with parcels; block scale (~50-110 parcels) measured ~0. A flat or\n"
          "  falling trend would falsify that story even if every sign is positive.\n")
    print(f"    {'parcels':>9}" + "".join(f"{f'd={d:.2f}':>10}" for d in BUDGETS))
    for ri in ids:
        cells = []
        for d in BUDGETS:
            got = _delta(out, ri, "128", f"{d:.2f}")
            cells.append(f"{got[1]:>+10.4f}" if got else f"{'--':>10}")
        print(f"    {int(out[ri]['parcels']):>9,}" + "".join(cells))  # type: ignore[arg-type]

    print("\n  speed, per region (uncapped minutes / capped minutes)")
    print(f"    {'region':<8}{'parcels':>9}{'uncapped min':>14}{'128':>9}{'256':>9}"
          f"{'128x':>9}{'256x':>9}")
    for ri in ids:
        arms = out[ri]["arms"]
        assert isinstance(arms, dict)
        if not all(lb in arms for lb in ("uncapped", "128", "256")):
            continue
        u, a, b = (float(arms[x]["secs"]) for x in ("uncapped", "128", "256"))
        print(f"    {ri:<8}{int(out[ri]['parcels']):>9,}{u / 60:>14.1f}"  # type: ignore[arg-type]
              f"{a / 60:>9.1f}{b / 60:>9.1f}{u / a:>8.1f}x{u / b:>8.1f}x")

    print("\n  A sign that does not flip across independent regions is the claim; one region's\n"
          "  magnitude is not. If perm improves everywhere at every budget the n=1 caveat closes;\n"
          "  if it flips, the original +0.0884 was one draw from a wide distribution -- which is\n"
          "  exactly what this method's known tie-break scatter would produce.")


if __name__ == "__main__":
    main()
