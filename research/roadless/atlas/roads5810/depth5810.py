"""How far in each clearing reaches on 5810 (owner 2026-10-26: "we're only clearing buildings at
the periphery"; "let's try J_2 as the objective to see if it clears further in"). A building's
depth is its home's escape time with nothing cleared (the plain baseline's, the model's own
"how far from the street"), as a quantile among the homes weighted by floor area: 0 beside the
street, 1 the deepest. Per method and target, of the cleared floor area: the share in the
deepest quarter and the shallowest quarter, and the median depth. Reads clear5810.py's frames
(SCRATCH/<key>.npz: `x_<fff>`, `u_base`); writes depth.json beside the page.

    PYTHONPATH=. uv run python research/roadless/atlas/roads5810/depth5810.py <key> ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_site import Site  # noqa: E402  (atlas/block_site.py)

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parents[3] / ".superpowers" / "sdd" / "2026-10-26-clearings5810"
BASE = Site.of("ZAF.9.3.1_1_5810").data      # the page's baseline (compute.baseline)
TARGETS = tuple(round(0.01 * i, 2) for i in range(1, 16))     # clear5810.TARGETS
OUT = HERE / "depth.json"


def depth_quantiles(u: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Each home's depth: the area-weighted share of homes with a shorter escape time (nan for a
    building with no trip of its own)."""
    ok = np.isfinite(u)
    o = np.argsort(u[ok])
    cw = np.cumsum(area[ok][o])
    q = np.full(len(u), np.nan)
    q[np.flatnonzero(ok)[o]] = (cw - 0.5 * area[ok][o]) / cw[-1]
    return q


def main(keys: list[str]) -> None:
    polys = shapely.from_wkb(np.load(BASE / "baseline.npz", allow_pickle=True)["polys"])
    area = shapely.area(polys)
    out = {}
    for key in keys:
        d = np.load(SCRATCH / f"{key}.npz")
        q = depth_quantiles(d["u_base"], area)
        rows = []
        for t in TARGETS:
            f = f"{int(round(t * 1000)):03d}"
            x = d[f"x_{f}"].astype(bool) & np.isfinite(q)
            a = area[x]
            o = np.argsort(q[x])
            med = float(q[x][o][np.searchsorted(np.cumsum(a[o]), 0.5 * a.sum())])
            rows.append(dict(target=t, deep=float(a[q[x] >= 0.75].sum() / a.sum()),
                             shallow=float(a[q[x] < 0.25].sum() / a.sum()), median=med,
                             no_trip=int((d[f"x_{f}"].astype(bool) & ~np.isfinite(q)).sum())))
        out[key] = rows
        print(key, " ".join(f"{r['target']:.2f}:{r['deep']:.2f}/{r['shallow']:.2f}/"
                            f"{r['median']:.2f}" for r in rows[::2]), flush=True)
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    OUT.write_text(json.dumps(old | out, indent=1) + "\n")


if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
