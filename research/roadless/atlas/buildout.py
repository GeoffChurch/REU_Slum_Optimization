"""A grown run's build-out on any block (owner 2026-10-28: the Demo page's second block), as the
5810 build-out page builds its own: the block's plain baseline (compute.baseline), the run's frames
at the slider's budgets D 0.01 .. 0.15 (networks.GrownSource: per budget the shortest prefix of the
run whose D reaches it, carved into one mesh and solved exactly under RUNTIME: compute.frames_of's
rasters, lenses, each home's escape time, the sheet), and its per-frame table. The block is the
command line's, resolved once into a Site (block_site.py); nothing below names one.

The table (Site.kept/<plan>.json, committed): per slider budget the frame (its stop: the run's
prefix length, D, road pieces and km, L_P, the conductance added C, and the run's recorded time
to that state, steps.parquet's cumulative t), the whole run's time and the machine, and the facts
the Demo's escape-time view is drawn on (the plain baseline's 99th percentile escape time, its
scale; explore_proto/build.py reads the rest from the Site's files).

    B=research/roadless/atlas/buildout.py              # from the repo root, PYTHONPATH=.
    uv run python $B baseline <block>
    uv run python $B frames <block> <plan>             # a growth_rows/<block>/<plan> run
    uv run python $B table <block> <plan> <machine>    # <machine>: where the run ran, in words
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "roads5810"))
import networks as nw  # noqa: E402  (the build-out page's sources; puts block_site on the path)
from block_site import Site  # noqa: E402

cp, render, gr = nw.cp, nw.render, nw.gr
PERCENTILE = 99          # the escape-time view's scale: the plain baseline's homes' percentile


def budget_of(plan: str) -> float:
    """A grown run's D budget, from its plan (ears.Plan or growth.Plan, by its name's head)."""
    import ears  # noqa: PLC0415  (its own study)
    spec = re.sub(r"\.v[0-9]+$", "", plan)
    return float(ears.Plan.of(spec).D if plan.startswith("ears.") else gr.Plan.of(spec).D)


def source(site: Site, plan: str) -> nw.GrownSource:
    src = nw.GrownSource(plan, budget=budget_of(plan), site=site)
    if not src.ready():
        raise SystemExit(f"{src.dir}: no finished run reaching D {src.budget:g}")
    return src


def baseline(site: Site) -> None:
    [b] = cp.common.build_blocks([site.bid])
    cp.baseline(b, data=site.data)


def frames(site: Site, plan: str) -> None:
    src = source(site, plan)
    [b] = cp.common.build_blocks([site.bid])
    src.frames(plan, b)


def table(site: Site, plan: str, machine: str) -> None:
    """The per-frame table (module docstring) into Site.kept/<plan>.json."""
    src = source(site, plan)
    rows = src.rows(plan)
    info = nw.info_of(plan, "slider", data=src.data)
    m_of = {f: m for m, f in render.frame_names(info).items()}
    steps = pd.read_parquet(src.dir / "steps.parquet").set_index("step")
    u = np.load(site.data / "baseline_times.npz")["u"]
    base = json.loads((site.data / "baseline.json").read_text())
    out: list[dict[str, Any]] = []
    for t, r in zip(render.SLIDER, rows, strict=True):
        assert r is not None                     # a grown run has a row at every budget
        m = m_of[r["f"]]
        out.append(dict(budget=t, frame=r["f"], stop=m, D=r["D"], roads=r["n"], km=r["km"],
                        L_P=r["LP"], C=r["C"], seconds=round(float(steps.t[m]), 1),
                        whole=bool(r.get("full", False))))
    doc = dict(block=site.bid, plan=plan, metric=base["metric"], machine=machine,
               run_seconds=round(float(steps.t.iloc[-1]), 1), run_steps=int(steps.index.max()),
               frames_seconds=round(float(info["t_total"]), 1),
               escape_scale=float(np.percentile(u[np.isfinite(u)], PERCENTILE)),
               escape_percentile=PERCENTILE, n_homes=int(np.isfinite(u).sum()),
               n_buildings=len(u), rows=out)
    site.kept.mkdir(parents=True, exist_ok=True)
    (site.kept / f"{plan}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{'budget':>6} {'stop':>4} {'D':>7} {'km':>5} {'L_P':>6} {'C':>7} {'s':>6}")
    for r in out:
        print(f"{r['budget']:6.2f} {r['stop']:4d} {r['D']:7.4f} {r['km']:5.2f} {r['L_P']:6.3f} "
              f"{r['C']:7d} {r['seconds']:6.1f}")


def main(argv: list[str]) -> None:
    match argv:
        case ["baseline", bid]:
            baseline(Site.of(bid))
        case ["frames", bid, plan]:
            frames(Site.of(bid), plan)
        case ["table", bid, plan, machine]:
            table(Site.of(bid), plan, machine)
        case _:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
