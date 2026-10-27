"""The uniform baselines of the 5810 build-out page (owner 2026-10-27: "a baseline pseudo-method
that just increases the conductance of everything (or e.g. just the open ground), so you can see
how much P decreases per unit of homogeneously increased conductance"): no building cleared, the
sheet conductance's boost scaled by 1 + eps, on every triangle (`all`: the sheet itself scaled,
the paved ways with it) or on the triangles off the paved ways (`open`: the ways' boost 1), eps
chosen so the conductance added (gap_clear.Model.sheet less the baseline's) meets each of TARGETS.
Each row is no method: it displaces no one, so the page draws it only where conductance added is
an axis, the reference a placement must beat. P does not fall as 1 / (1 + eps) even under `all`:
the leaks to ground (gap_exits) are per cell and fixed.

    S=research/roadless/atlas/roads5810/uniform.py     # from the repo root, PYTHONPATH=.
    uv run python $S rows                              # SCRATCH/uniform.json
    uv run python $S page                              # UDATA into the page
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "research" / "roadless"))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_factor  # noqa: E402
import metrics  # noqa: E402

BID = "ZAF.9.3.1_1_5810"
SCRATCH = ROOT / ".superpowers/sdd/2026-10-27-uniform5810"
TARGETS = (2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6)   # m^2: the conductance added


def rows() -> None:
    t0 = time.time()
    [b] = common.build_blocks([BID])
    m = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU()).clearing(
        b, demand=common.PopulationDemand())
    k0 = np.zeros(m.n)
    st0 = m.factors(k0)
    base = m.sheet(st0)
    on = dict(all=np.ones(m.fx.fm.nt, dtype=bool), open=m.b_ways == 1.0)
    out: dict = dict(base=base, P0=m.base.P, variants={})  # type: ignore[type-arg]
    for name, sel in on.items():
        held = float(m.tri_area[sel] @ (st0.kt * st0.b)[sel])   # the sheet the variant scales
        vrows = []
        for C in TARGETS:
            eps = C / held
            st = m.boosted_state(k0, np.where(sel, m.b_ways * (1.0 + eps), m.b_ways))
            ln = m.lenses_of(st)
            vrows.append(dict(C=m.sheet(st) - base, eps=eps, L_P=1.0 - ln.P / m.base.P,
                              L_Pq=1.0 - ln.Pq / m.base.Pq, L_J2=1.0 - ln.J2 / m.base.J2))
            print(f"{name} C {vrows[-1]['C']:.0f} (eps {eps:.4g}): L_P {vrows[-1]['L_P']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        out["variants"][name] = dict(held=held, rows=vrows)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "uniform.json").write_text(json.dumps(out, indent=1))


def page() -> None:
    """UDATA (per variant its rows' C and L_P, the page's `C` and `LP`) into the page, after
    CDATA."""
    path = HERE / "index.html"
    html = path.read_text()
    u = json.loads((SCRATCH / "uniform.json").read_text())
    text = json.dumps({name: [dict(C=round(r["C"]), LP=round(r["L_P"], 4)) for r in v["rows"]]
                       for name, v in u["variants"].items()}, separators=(",", ":"))
    line = f"var UDATA = {text};\n"
    if "var UDATA = " in html:
        html = re.sub(r"var UDATA = \{.*?\};\n", lambda _: line, html, count=1)
    else:
        html = re.sub(r"(var CDATA = \{.*?\};\n)", lambda g: g[1] + line, html, count=1)
    path.write_text(html)
    print("UDATA into the page", flush=True)


if __name__ == "__main__":
    arg, *rest = sys.argv[1:]
    if arg == "rows" and not rest:
        rows()
    elif arg == "page" and not rest:
        page()
    else:
        raise SystemExit(f"{sys.argv[1:]}: rows | page")
