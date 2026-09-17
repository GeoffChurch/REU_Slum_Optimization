"""How far can the cheap pre-filter be cut before the screen's top-k changes?

`proxy_keep_pct` is the ONLY screen knob that saves time. `DenseCompactScreen` runs
proxy -> keep the top `proxy_keep_pct`% by proxy -> Voronoi + BFS peel over those survivors ->
metric.fine -> gate. The peel is the expensive stage and the gate runs AFTER it, so the gate's
width (the absolute floors in `reblock.metric`) costs no compute at all -- widen or narrow it and
the same blocks were peeled either way. Cutting `proxy_keep_pct` is what removes work.

The knob only exists for `needs_peel=True` metrics. `depth_density_proxy` (the shipped default)
and `density_compactness` score straight from the free kblock columns and never reach this branch,
so for them the setting is inert.

What this measures: for each retention, whether the final ranked top-1/5/15 is the same as at the
shipped 50%. A tighter pre-filter can only LOSE blocks -- a block that is not peeled cannot be
scored, and a block with a modest proxy but a large true peel depth is exactly what a tight filter
drops. That is the failure mode, and it is why this is measured per city rather than argued.

    pixi run python -m scripts.proxy_keep_retention
    pixi run python -m scripts.proxy_keep_retention --cities capetown --pcts 1,2,5

NOT a wall-clock benchmark. Every peel in a developed checkout is an L2 cache hit, so the timings
such a sweep reports are lookup costs, not work. The honest cost unit is blocks peeled, which is
`proxy_keep_pct`% of the eligible corpus exactly; a cold peel measured 0.17 s/block on median-size
Cape Town blocks, across a 16-way fork pool.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

PCTS = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
KS = (1, 5, 15)
BASELINE = 50.0  # the shipped conf/config.yaml value every row is compared against
VARIANTS = ("depth", "depth_density")  # the only two needs_peel metrics with example configs


def selection(variant: str, city: str, pct: float) -> list[str]:
    """The screen's ranked block_ids under one retention, built from the SHIPPED config objects.

    Composed through hydra rather than constructed by hand so this measures the screen the
    pipeline actually runs -- same metric, same gate, same counts strategy.
    """
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(
            config_name="compare_config",
            overrides=[f"+example={variant}", f"data={city}_full", f"proxy_keep_pct={pct}"],
        )
    return list(instantiate(cfg.screen).select(instantiate(cfg.data)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cities", default="capetown,nairobi")
    ap.add_argument("--pcts", default=",".join(str(p) for p in PCTS))
    args = ap.parse_args()
    pcts = [float(p) for p in args.pcts.split(",")]
    if BASELINE not in pcts:
        pcts.append(BASELINE)

    print(f"{'city':<10}{'variant':<15}" + "".join(f"{f'{p:g}%':>14}" for p in sorted(pcts)))
    worst: dict[int, float] = {}
    for city in args.cities.split(","):
        for variant in VARIANTS:
            res = {p: selection(variant, city, p) for p in sorted(pcts)}
            base = res[BASELINE]
            cells = []
            for p in sorted(pcts):
                marks = ""
                for k in KS:
                    if res[p][:k] == base[:k]:
                        marks += "="
                    elif set(res[p][:k]) == set(base[:k]):
                        marks += "~"
                    else:
                        marks += "X"
                        worst[k] = max(worst.get(k, 0.0), p)
                cells.append(f"{len(res[p]):>7}:{marks}")
            print(f"{city:<10}{variant:<15}" + "".join(f"{c:>14}" for c in cells))

    print("\ncell = blocks peeled : top-1/top-5/top-15 against the 50% baseline")
    print("  '=' identical prefix   '~' same set, different order   'X' differs")
    for k in KS:
        if k in worst:
            print(f"  top-{k} first differs at or below {worst[k]:g}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
