"""The matched-displacement comparison, as a RE-ANALYSIS of saved road lists.

`region_anchor_cap.py`'s headline compared arms at "displacement budget 0.10" on a region where the
networks displace 0.0115-0.0193. `prefix_to_displacement` returns all roads when a budget is
unreachable -- documented, silent, no error -- so nothing was ever truncated and the comparison was
road-count-matched. The capped arm spent 68% more displacement than uncapped and was not charged
for it, which is where its apparent quality win came from.

The fix is not a bigger budget but a *reachable* one, and reachability is a property of the arms,
not something to pick in advance. So: read each arm's FULL road list, find the largest displacement
every arm in that region can actually reach, and evaluate all of them at fractions of it. Absolute
budgets carried over from block scale (0.05-0.20) are meaningless here and 0.005-0.02 is the real
band -- but it differs per region, which is exactly why it is computed rather than configured.

This runs no greedy. It exists because `region_cap_replicate.py` persists full road lists, so every
future budget question is minutes of evaluation instead of hours of search. The WKT round-trip drops
the `width_m` column; every run here used `DEFAULT_ROAD_WIDTH_M`, so it is restored on load.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
from shapely import wkt

from reblock.budget import building_radii, displacement, prefix_to_displacement
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.eval.access_burden import burden
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, permeability
from scripts.perf import region_pool

SRC = Path("scripts/perf/region_cap_replicate.json")
OUT = Path("scripts/perf/region_cap_matched.json")
FRACTIONS = (0.25, 0.50, 0.75, 1.00)   # of the largest displacement EVERY arm in the region reaches
ARMS = ("uncapped", "128", "256")


def _gdf(wkts: list[str], block: Block) -> gpd.GeoDataFrame:
    geoms = [wkt.loads(s) for s in wkts]
    return gpd.GeoDataFrame({"width_m": [DEFAULT_ROAD_WIDTH_M] * len(geoms)},
                            geometry=geoms, crs=block.crs)


def main() -> int:
    if not SRC.exists():
        print(f"{SRC} not found -- run scripts.perf.region_cap_replicate first")
        return 1
    data = json.load(SRC.open())
    pool = region_pool.blocks(6)
    out: dict[str, dict[str, object]] = {}

    for ri in sorted(data, key=lambda r: int(data[r]["parcels"])):
        arms = data[ri]["arms"]
        if not all(a in arms for a in ARMS):
            print(f"  region {ri}: incomplete ({sorted(arms)}) -- skipped", flush=True)
            continue
        block = pool[int(ri)]
        n = len(block.parcels)
        adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
        radii = building_radii(block.building_points)
        nb = len(block.building_points)
        b0 = burden(parcel_access_layers(block, None, tol=STREET_TOL, adj=adj,
                                         unreached_depth=n + 1))
        roads = {a: _gdf(arms[a]["roads_wkt"], block) for a in ARMS}
        reach = {a: displacement(block.building_points, radii, roads[a]) / nb for a in ARMS}
        dmax = min(reach.values())
        print(f"\n  region {ri}: {n:,} parcels; arm displacement reach "
              + ", ".join(f"{a}={reach[a]:.4f}" for a in ARMS)
              + f"  -> matching band 0..{dmax:.4f}", flush=True)

        rec: dict[str, object] = {"parcels": n, "reach": reach, "dmax": dmax, "at": {}}
        at: dict[str, dict[str, dict[str, float]]] = {}
        for f in FRACTIONS:
            d = dmax * f
            row: dict[str, dict[str, float]] = {}
            for a in ARMS:
                pre = prefix_to_displacement(block, roads[a], radii, d)
                if len(pre) == 0:
                    continue
                b1 = burden(parcel_access_layers(block, pre, tol=STREET_TOL, adj=adj,
                                                 unreached_depth=n + 1))
                row[a] = {"burden_red": (1.0 - b1 / b0) if b0 > 0 else 0.0,
                          "perm": float(permeability(block, pre)),
                          "road_m": float(pre.geometry.length.sum()),
                          "n_roads": float(len(pre))}
            at[f"{f:.2f}"] = row
            if len(row) == len(ARMS):
                print(f"    d={d:.4f} ({f:.0%})  "
                      + "  ".join(f"{a}: b={row[a]['burden_red']:.4f} p={row[a]['perm']:.4f}"
                                  for a in ARMS), flush=True)
        rec["at"] = at
        out[ri] = rec
        OUT.write_text(json.dumps(out, indent=1))

    if not out:
        print("nothing to analyse")
        return 1
    print("\n  Recorded. Analysis lives in one place:")
    print("    pixi run python -m scripts.perf.region_cap_report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
