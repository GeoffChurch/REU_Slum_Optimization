"""How much of `greedy_arterial`'s outcome is decided by ARBITRARY TIE-BREAKS?

Discovered accidentally. An incremental reformulation of the displacement cost agreed with the
original to 1.14e-10 -- and produced a different network on 29% of runs, moving burden reduction by
up to 11 POINTS on one block. A 1e-10 perturbation should not move a headline number by 11 points
unless the argmax is sitting on near-ties.

If that is right it is a property of the METHOD, not of the optimisation: any perturbation would
surface it -- a different shapely build, a different platform, a different candidate order. The
published numbers would then be one arbitrary draw from a distribution, and nobody has ever looked
at the spread of that distribution.

## A first version measured the wrong thing

It shuffled the ORDER candidates are scored in, on the reasoning that `max` returns the first
maximal element so reordering is the tie-break knob. Result: **zero spread on 8/8 blocks x 6
orderings** -- and the probe was verified live (the shuffle reorders in place; `_candidate_chords`
was intercepted once per step), so that null is real.

It is real and irrelevant. Shuffling only decides EXACT ties, and there are none. What the
displacement reformulation perturbed was the gain VALUES, which reorders candidates that are NEARLY
tied -- and a near-tie is resolved by value, deterministically, regardless of order. Two different
sensitivities; the first version measured the one that is not the phenomenon.

## The experiment

Perturb each candidate's gain by a relative epsilon of the same order as the discrepancy that
started this (1e-10), with several seeds, and measure the spread of the reported objectives. That is
exactly the perturbation `displacement_fast` applies, minus the speedup -- so any spread is
attributable to near-ties in the argmax and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

import reblock.methods.arterial.engines as art
from reblock.budget import prefix_to_displacement
from reblock.data.pools import capetown_pool, evenly_spaced, load_pools
from reblock.derive.access import (
    STREET_TOL,
    ParcelAdjacency,
    parcel_access_layers,
    past_every_parcel,
)
from reblock.eval.access_burden import burden
from reblock.methods.arterial import (
    Access,
    Displacement,
    ExactEngine,
    GreedyArterialReblocker,
    SnapToBoundary,
)
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    EgressContext,
    PermeabilityParams,
    permeability,
)

SEEDS = [None, 1, 2, 3, 4, 5]      # None = unperturbed (the shipped answer)
N_BLOCKS = 8
OUT = Path("scripts/perf/tie_sensitivity.json")


EPS = 1e-10          # the order of the displacement-reformulation discrepancy (measured 1.14e-10)

# Module-level, not a closure: `_greedy_arterials` hands `eval_candidate` to a fork pool, and
# `multiprocessing` pickles it BY QUALIFIED NAME. A local closure fails with
# "Can't get local object '_patch.<locals>.jittered'". The seed rides along as a module global,
# which forked children inherit at fork time.
_JITTER_SEED: int | None = None
_ORIG_EVAL = art.eval_candidate


def _jittered(chord: LineString) -> tuple[float, BaseGeometry | None]:
    gain, real = _ORIG_EVAL(chord)
    if _JITTER_SEED is None or not np.isfinite(gain) or gain == 0.0:
        return gain, real
    # deterministic per (seed, candidate), so a rerun with the same seed reproduces exactly
    h = abs(hash((_JITTER_SEED, round(float(chord.length), 9),
                  round(float(chord.centroid.x), 6))))
    u = (np.random.default_rng(h % (2**32)).random() - 0.5) * 2.0
    return gain * (1.0 + EPS * u), real


def _patch(seed: int | None) -> None:
    """Perturb each candidate's GAIN by a relative `EPS`; None restores the shipped scorer."""
    global _JITTER_SEED
    _JITTER_SEED = seed
    art.eval_candidate = _ORIG_EVAL if seed is None else _jittered


def main() -> None:
    pools = load_pools(capetown_pool(Path("conf")))
    blocks = pools.blocks
    counts = [float(len(b.parcels)) for b in blocks]
    sel = [i for i in pools.recipients if len(blocks[i].parcels) <= 110]
    rows: dict[str, dict[str, list[float]]] = {}
    for i in evenly_spaced(sorted(sel), counts, N_BLOCKS):
        b = blocks[i]
        adjacency = ParcelAdjacency.of(b, STREET_TOL)
        ctx = EgressContext(adjacency, PermeabilityParams())
        b0 = burden(parcel_access_layers(adjacency, None, unreached=past_every_parcel))
        rec: dict[str, list[float]] = {"burden_red": [], "perm": [], "road_m": [], "n_roads": []}
        for seed in SEEDS:
            _patch(seed)
            m = GreedyArterialReblocker(realizer=SnapToBoundary(lam=2.0), objective=Access(),
                                        cost=Displacement(), workers=8, max_roads=8, n_anchors=32,
                                        top_k=8, road_width_m=DEFAULT_ROAD_WIDTH_M,
                                        engine=ExactEngine(), max_anchors=0)
            r = m.propose(b).roads
            if r is None or len(r) == 0:
                continue
            pre = prefix_to_displacement(b, r, 0.10)
            if len(pre) == 0:
                continue
            b1 = burden(parcel_access_layers(adjacency, pre, unreached=past_every_parcel))
            rec["burden_red"].append((1.0 - b1 / b0) if b0 > 0 else 0.0)
            rec["perm"].append(float(permeability(ctx, pre)))
            rec["road_m"].append(float(pre.geometry.length.sum()))
            rec["n_roads"].append(float(len(r)))
        _patch(None)
        if len(rec["burden_red"]) >= 3:
            rows[b.block_id] = rec
            sp = max(rec["burden_red"]) - min(rec["burden_red"])
            print(f"  {b.block_id:<22} n={len(rec['burden_red'])}  "
                  f"burden_red {min(rec['burden_red']):.4f}..{max(rec['burden_red']):.4f} "
                  f"(spread {sp:+.4f})", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))

    print(f"\n{'=' * 74}\nNEAR-TIE SENSITIVITY -- {len(rows)} blocks "
          f"x {len(SEEDS)} gain perturbations\n")
    print(f"  Each candidate gain multiplied by (1 +- {EPS:.0e}) -- the size of the\n"
          "  displacement-reformulation discrepancy. Nothing else changed.\n")
    for metric in ("burden_red", "perm", "road_m"):
        spreads = np.array([max(v[metric]) - min(v[metric]) for v in rows.values()])
        rels = np.array([(max(v[metric]) - min(v[metric])) / max(abs(np.mean(v[metric])), 1e-9)
                         for v in rows.values()])
        print(f"  {metric:<12} spread across seeds: median {np.median(spreads):.4f}, "
              f"max {spreads.max():.4f}   (relative: median {np.median(rels):.1%}, "
              f"max {rels.max():.1%})")
    identical = sum(1 for v in rows.values() if len(set(np.round(v["burden_red"], 12))) == 1)
    print(f"\n  blocks where every perturbation gave the SAME burden reduction: "
          f"{identical}/{len(rows)}")


if __name__ == "__main__":
    main()
