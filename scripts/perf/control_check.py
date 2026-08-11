"""Is the A/B's CONTROL arm actually the shipped greedy?

`shortlist_greedy.greedy_shortlist` re-states `_greedy_arterials`' step loop so an injected
selector can cut the candidate list in the middle of it. With `ScoreAll()` it should reduce to the
shipped function
exactly -- but "should" is doing real work there: the two now have separate copies of a dozen
per-step setup lines, and dropping one (`committed_disp`, `base_val`, the `step` context) changes
scores silently rather than crashing.

If the control arm is not the shipped greedy, every deviation the A/B reports is measured against
the wrong baseline and the whole comparison is void. So: same block, both functions, geometry
compared WKT for WKT.
"""
from __future__ import annotations

import sys

from reblock.methods.arterial.engines import _greedy_arterials
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from scripts.pair_matrix import evenly_spaced, load_pools
from scripts.perf.selectors import ScoreAll
from scripts.perf.shortlist_greedy import greedy_shortlist

N_BLOCKS = 3
MAX_ROADS = 5


def main() -> int:
    pools = load_pools()
    blocks = pools.blocks
    counts = [float(len(b.parcels)) for b in blocks]
    sel = [i for i in pools.recipients if len(blocks[i].parcels) <= 80]
    bad = 0
    for i in evenly_spaced(sorted(sel), counts, N_BLOCKS):
        b = blocks[i]
        kw = dict(mode="buildable", objective="access", cost="displacement",
                  half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0, workers=8, max_roads=MAX_ROADS)
        want = _greedy_arterials(b, **kw)                      # type: ignore[arg-type]
        got = greedy_shortlist(b, selector=ScoreAll(), **kw)    # type: ignore[arg-type]
        w = [g.wkt for g in want.geometry]
        p = [g.wkt for g in got.geometry]
        ok = w == p
        bad += not ok
        print(f"  {b.block_id:<22} n={len(b.parcels):<4} shipped {len(w):>2} roads, "
              f"control {len(p):>2} roads  ->  {'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
        if not ok:
            for j, (a, c) in enumerate(zip(w, p, strict=False)):
                if a != c:
                    print(f"      first divergence at road {j}:\n        shipped {a[:110]}\n"
                          f"        control {c[:110]}")
                    break
    verdict = ("PASS -- control arm IS the shipped greedy" if not bad
               else f"FAIL -- {bad} blocks differ")
    print(f"\n  {verdict}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
