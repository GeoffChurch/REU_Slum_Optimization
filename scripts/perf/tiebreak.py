"""Which TIE-BREAK should the greedy use when candidates have exactly equal gain?

`_best_candidate` already breaks ties -- on `real.wkt < best_real.wkt`, lexicographic order over the
coordinate STRING. That is deterministic and order-independent (which is why shuffling candidate
order changes nothing) and geometrically meaningless: it picks whichever road's text sorts first.

The ties are not rare or degenerate. On the worst block, 6 of 8 greedy steps have an exact top-gain
tie, and at step 1 NINE candidates share a bit-identical gain while realizing SIX different roads up
to 38.8 m apart (`notes/2026-08-09-greedy-arterial-is-near-tie-sensitive.md`). They arise because
the access objective is a sum of squared INTEGER depths, so genuinely different networks routinely
achieve the identical improvement -- ties are structural here, not incidental.

So the choice is real and currently made by string sort. This measures three rules:

    wkt        today: lexicographic on the WKT string          (arbitrary, the baseline)
    shortest   prefer the shorter road at equal gain           (spend less, leave budget)
    longest    prefer the longer road at equal gain            (bank more absolute benefit)

Equal GAIN means equal benefit-per-cost, so at a tie the shorter road buys proportionally less
benefit for proportionally less cost. Which is better for a budgeted greedy is not obvious a priori
-- hence measuring. WKT remains the final fallback in every rule, so all three stay deterministic
and order-independent.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from shapely.geometry.base import BaseGeometry

import reblock.methods.arterial as art
from reblock.budget import building_radii, prefix_to_displacement
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.eval.access_burden import burden
from reblock.methods.arterial import GreedyArterialReblocker
from reblock.permeability import permeability
from scripts.pair_matrix import evenly_spaced, load_pools

RULES = ("wkt", "shortest", "longest")
N_BLOCKS = 12
OUT = Path("scripts/perf/tiebreak.json")
_RULE = "wkt"


def _best(results: Iterable[tuple[float, BaseGeometry | None]]
          ) -> tuple[float, BaseGeometry | None]:
    """`_best_candidate` with a pluggable tie-break. Same `(0.0, None)` init and same gating, so a
    non-positive gain can still never win (that IS the termination condition)."""
    best_gain, best_real = 0.0, None
    for gain, real in results:
        if gain > best_gain:
            best_gain, best_real = gain, real
        elif (best_real is not None and real is not None and gain == best_gain):
            if _RULE == "shortest":
                better = (real.length, real.wkt) < (best_real.length, best_real.wkt)
            elif _RULE == "longest":
                better = (-real.length, real.wkt) < (-best_real.length, best_real.wkt)
            else:
                better = real.wkt < best_real.wkt
            if better:
                best_gain, best_real = gain, real
    return best_gain, best_real


def main() -> None:
    art._best_candidate = _best                       # type: ignore[assignment]
    pools = load_pools()
    blocks = pools.blocks
    counts = [float(len(b.parcels)) for b in blocks]
    sel = [i for i in pools.recipients if len(blocks[i].parcels) <= 110]
    rows: dict[str, dict[str, dict[str, float]]] = {}
    global _RULE
    for i in evenly_spaced(sorted(sel), counts, N_BLOCKS):
        b = blocks[i]
        adj = parcel_adjacency(list(b.parcels.geometry), STREET_TOL)
        radii = building_radii(b.building_points)
        n = len(b.parcels)
        b0 = burden(parcel_access_layers(b, None, tol=STREET_TOL, adj=adj, unreached_depth=n + 1))
        rec: dict[str, dict[str, float]] = {}
        for rule in RULES:
            _RULE = rule
            m = GreedyArterialReblocker(mode="buildable", objective="access",
                                        cost="displacement", workers=8, max_roads=8)
            r = m.propose(b).roads
            if r is None or len(r) == 0:
                continue
            pre = prefix_to_displacement(b, r, radii, 0.10)
            if len(pre) == 0:
                continue
            b1 = burden(parcel_access_layers(b, pre, tol=STREET_TOL, adj=adj,
                                             unreached_depth=n + 1))
            rec[rule] = {"burden_red": (1.0 - b1 / b0) if b0 > 0 else 0.0,
                         "perm": float(permeability(b, pre)),
                         "road_m": float(pre.geometry.length.sum())}
        if len(rec) == len(RULES):
            rows[b.block_id] = rec
            print(f"  {b.block_id:<22}" + "  ".join(
                f"{k}={v['burden_red']:.4f}" for k, v in rec.items()), flush=True)
    OUT.write_text(json.dumps(rows, indent=1))

    print(f"\n{'=' * 72}\nTIE-BREAK COMPARISON -- {len(rows)} blocks\n")
    print(f"  {'rule':<12}{'burden_red':>12}{'perm':>10}{'road_m':>10}{'beats wkt':>12}")
    for rule in RULES:
        br = np.array([v[rule]["burden_red"] for v in rows.values()])
        pm = np.array([v[rule]["perm"] for v in rows.values()])
        rm = np.array([v[rule]["road_m"] for v in rows.values()])
        base = np.array([v["wkt"]["burden_red"] for v in rows.values()])
        wins = int((br > base).sum())
        ties = int((br == base).sum())
        print(f"  {rule:<12}{np.median(br):>12.4f}{np.median(pm):>10.4f}{np.median(rm):>10.1f}"
              f"{wins:>7}/{len(br) - ties:<4}")
    ident = sum(1 for v in rows.values()
                if len({round(v[r]["burden_red"], 12) for r in RULES}) == 1)
    print(f"\n  blocks where all three rules agree: {ident}/{len(rows)}")


if __name__ == "__main__":
    main()
