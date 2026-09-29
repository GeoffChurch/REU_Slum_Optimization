"""Monotonicity over random nested road sets (both rules), and the crossing diagnostic.

    PYTHONPATH=. pixi run python research/roadless/mono_cross.py mono <block idx> <h>
    PYTHONPATH=. pixi run python research/roadless/mono_cross.py cross <block idxs> <h>
"""
import sys, time; sys.path.insert(0, 'research/roadless')
import numpy as np, common, lifted
from reblock.derivations import propose
from reblock.budget import prefix_to_displacement


def mono(i: int, h: float) -> None:
    blocks = common.build_blocks(common.recipients())
    arms = common.arms()
    b = blocks[i]
    rng = np.random.default_rng(0)
    for rule in (common.Carve(), common.Obliterate()):
        sc = common.Scorer(b, h, lifted.Params(ell_m=3.0, K=16), rule=rule)
        worst = 0.0
        steps = 0
        for n in ("cycle_native", "resistance_lp"):
            r = propose(arms[n], b).roads.reset_index(drop=True)
            for trial in range(3):
                order = rng.permutation(len(r))
                cuts = np.unique(np.linspace(0, len(r), 9).astype(int))
                prev = sc.P0
                for c in cuts[1:]:
                    P = sc.P_free(sc.free_of(r.iloc[order[:c]]))
                    worst = max(worst, (P - prev) / prev)
                    prev = P
                    steps += 1
        print(f"{b.block_id} h={h} {rule.name}: {steps} nested steps, largest relative INCREASE "
              f"{worst:.2e} (must be <= ~1e-8)", flush=True)


def cross(idx: list[int], h: float) -> None:
    blocks = common.build_blocks(common.recipients())
    arms = common.arms()
    p = lifted.Params(ell_m=3.0, K=16)
    for i in idx:
        b = blocks[i]
        sc = common.Scorer(b, h, p)
        sol = lifted.solve(sc.grid, sc.free0, sc.f, p)
        d0 = lifted.crossing(sc.free0, sc.grid.ground, sol, p)
        r = prefix_to_displacement(b, propose(arms["cycle_native"], b).roads, 0.10)
        fr = sc.free_of(r)
        d1 = lifted.crossing(fr, sc.grid.ground, lifted.solve(sc.grid, fr, sc.f, p), p)
        fmt = lambda d: " ".join(f"{k} {v:.3f}" for k, v in d.items())
        print(f"{b.block_id} h={h} no roads: {fmt(d0)} | cycle_native@10%: {fmt(d1)}", flush=True)


def cross_channel() -> None:
    from shapely.geometry import LineString, Polygon
    p = lifted.Params(ell_m=3.0, K=16)
    for a in (0, 20, 45):
        th = np.radians(a); e = np.array([np.cos(th), np.sin(th)]); n = np.array([-e[1], e[0]])
        W, Lc = 4.0, 30.0
        c = [-n * W / 2, e * Lc - n * W / 2, e * Lc + n * W / 2, n * W / 2]
        grid = lifted.Grid.of(Polygon(c), [], [LineString([c[1], c[2]])], 0.5)
        f = (grid.mask_of(LineString([c[0], c[3]]).buffer(1.0)) & grid.inside & ~grid.ground).astype(float)
        f /= f.sum()
        free = grid.inside
        sol = lifted.solve(grid, free, f, p)
        print(f"single stream, channel at {a} deg:", lifted.crossing(free, grid.ground, sol, p), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "mono":
        mono(int(sys.argv[2]), float(sys.argv[3]))
    elif sys.argv[1] == "channel":
        cross_channel()
    else:
        cross([int(a) for a in sys.argv[2].split(",")], float(sys.argv[3]))
