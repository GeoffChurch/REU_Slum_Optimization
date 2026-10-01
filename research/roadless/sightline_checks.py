"""Sightline conductance: (1) a straight channel's conduction vs its width, Uniform against
Sightline (should grow faster than W); (2) monotonicity on a real block (freeing buildings never
raises P); (3) operator build time.

    PYTHONPATH=. pixi run python research/roadless/sightline_checks.py [block idx]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402

ALONGS = [lifted.Uniform(), lifted.Sightline(1.0), lifted.Sightline(3.0), lifted.Sightline(10.0)]


def channel(width: float, length: float = 60.0, angle_deg: float = 0.0, h: float = 0.5):
    """A W x L channel in a solid block; the street is one short end, demand 1 at the other."""
    box = shapely.box(-40, -40, 40 + length, 40)
    ch = shapely.box(0, -width / 2, length, width / 2)
    rot = lambda g: shapely.affinity.rotate(g, angle_deg, origin=(0, 0))  # noqa: E731
    solid = rot(box.difference(ch))
    street = rot(shapely.LineString([(0, -width), (0, width)]))
    return lifted.Grid.of(rot(box), [solid], [street], h)


def main(idx: int) -> None:
    print("(1) channel L 60 m: P x W (Uniform ~ const; Sightline should fall as W grows)")
    for ang in (0.0, 30.0):
        for along in ALONGS:
            p = lifted.Params(ell_m=3.0, K=8, along=along)
            out = []
            for W in (1.0, 2.0, 4.0, 8.0):
                g = channel(W, angle_deg=ang)
                f = np.zeros(g.inside.shape)
                far = (g.ff0 > 0.99) & (np.hypot(*(g.xy - np.array(
                    shapely.affinity.rotate(shapely.Point(58, 0), ang, origin=(0, 0)).coords[0])
                ).transpose(2, 0, 1)) < W / 2)
                f[far] = 1.0 / far.sum()
                out.append(lifted.solve(g, g.ff0, f, p).P * W)
            print(f"  angle {ang:4.0f}  {along.name:5s}  " + "  ".join(f"W{w:g}: {v:8.2f}"
                  for w, v in zip((1, 2, 4, 8), out, strict=True)), flush=True)
    b = common.build_blocks(common.recipients())[idx]
    print(f"\n(2) monotonicity on {b.block_id} (n={len(b.buildings)}): P after removing nested random "
          "sets of buildings (must not rise)")
    rng = np.random.default_rng(0)
    for along in ALONGS[1:]:
        p = lifted.Params(ell_m=3.0, K=8, along=along)
        sc = common.Scorer(b, 0.5, p)
        order = rng.permutation(len(sc.polys))
        lab = sc.grid.label_sub(sc.polys)
        prev, rises = sc.P0, 0.0
        for n in (1, 3, 10, 30, 60):
            free = (sc.grid.isub & (~sc.grid.bsub | np.isin(lab, order[:n]))).mean(axis=-1)
            t = time.time()
            P = sc.P_free(free)
            rises = max(rises, (P - prev) / prev)
            prev = P
        print(f"  {along.name:5s}  P0 {sc.P0:9.2f} -> {prev:9.2f}  largest relative rise {rises:+.1e}"
              f"  (one solve {time.time() - t:.1f}s)", flush=True)
    t = time.time()
    lifted.operator(sc.grid.ff0, sc.grid.ground, sc.grid.h, lifted.Params(3.0, 8, ALONGS[0]))
    t0 = time.time() - t
    t = time.time()
    lifted.operator(sc.grid.ff0, sc.grid.ground, sc.grid.h, lifted.Params(3.0, 8, ALONGS[2]))
    print(f"\n(3) operator build: Uniform {t0:.1f}s, Sightline {time.time() - t:.1f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 110)
