"""Summarize `null_model.json`: does the first-order ranking beat a uniform-random subsample?

Separate from `null_model.py` because that script crashed in its own summary after writing the
results (it compared 40 pooled random draws against 8 exact values and numpy refused to broadcast).
The measurements were already on disk, so this re-derives the summary without re-running ten
minutes of greedies -- and keeps the analysis independent of the run, which is where it belongs.

The comparison that matters is per block: where does `FirstOrder(k)` sit inside the spread of
`RandomSample(k)` over seeds? Pooling all draws into one median hides exactly that.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC = Path("scripts/perf/null_model.json")
KS = (128, 32)
SEEDS = (1, 2, 3, 4, 5)


def main() -> None:
    rows = json.loads(SRC.read_text())
    ex = np.array([v["exact"]["burden_red"] for v in rows.values()])
    print(f"\n{'=' * 88}\nDOES THE RANKING BEAT A COIN FLIP? -- {len(rows)} blocks\n")
    print(f"  {'arm':<20}{'burden_red':>12}{'perm':>9}{'secs':>8}{'vs exact':>11}"
          f"{'beats exact':>13}")

    def show(label: str, key: str | None = None, pooled: list[str] | None = None) -> None:
        names = pooled if pooled is not None else [key]
        b = np.concatenate([[v[n]["burden_red"] for v in rows.values()] for n in names])
        p = np.concatenate([[v[n]["perm"] for v in rows.values()] for n in names])
        s = np.concatenate([[v[n]["secs"] for v in rows.values()] for n in names])
        # tile the baseline to match pooled draws -- the bug that killed the original summary
        base = np.tile(ex, len(names))
        print(f"  {label:<20}{np.median(b):>12.4f}{np.median(p):>9.4f}{np.median(s):>8.1f}"
              f"{np.median(b) - np.median(ex):>+11.4f}{(b > base).sum():>9}/{len(b):<3}")

    show("exact", "exact")
    for k in KS:
        show(f"fo-{k}", f"fo-{k}")
        show(f"rand-{k} (x{len(SEEDS)})", pooled=[f"rand-{k}-s{s}" for s in SEEDS])

    print("\n  PER BLOCK -- where the ranking sits in the random arm's own spread\n")
    print(f"  {'block':<24}{'k':>4}{'exact':>9}{'fo':>9}{'rand mean':>11}{'rand max':>10}"
          f"{'fo pctile':>11}")
    for k in KS:
        pct = []
        for bid, v in rows.items():
            fo = v[f"fo-{k}"]["burden_red"]
            rnd = np.array([v[f"rand-{k}-s{s}"]["burden_red"] for s in SEEDS])
            q = float((rnd < fo).mean())
            pct.append(q)
            if k == KS[0]:
                print(f"  {bid:<24}{k:>4}{v['exact']['burden_red']:>9.4f}{fo:>9.4f}"
                      f"{rnd.mean():>11.4f}{rnd.max():>10.4f}{q:>10.0%}")
        above_mean = sum(1 for bid, v in rows.items()
                         if v[f"fo-{k}"]["burden_red"]
                         > np.mean([v[f"rand-{k}-s{s}"]["burden_red"] for s in SEEDS]))
        above_max = sum(1 for bid, v in rows.items()
                        if v[f"fo-{k}"]["burden_red"]
                        > max(v[f"rand-{k}-s{s}"]["burden_red"] for s in SEEDS))
        print(f"\n    k={k}: ranking above the random MEAN on {above_mean}/{len(rows)} blocks, "
              f"above every random draw on {above_max}/{len(rows)};"
              f"\n         median percentile within the random spread: {np.median(pct):.0%}\n")

    print("  A ranking that were decoration would sit at the 50th percentile of the random draws.")


if __name__ == "__main__":
    main()
