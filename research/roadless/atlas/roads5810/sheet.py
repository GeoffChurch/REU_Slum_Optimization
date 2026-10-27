"""The conductance each frame of the 5810 build-out page adds (owner 2026-10-27: the roads' and the
clearings' costs in one unit, a monotone function of the conductance added, the identity, since a
footprint's resistance may be infinite): per frame the sheet conductance its state holds
(gap_clear.Model.sheet: per triangle area x factor x boost, in m^2 of open ground) less its own
mesh's baseline's. A clearing opens its footprints and the shy zones they alone held shut; a road
opens what its corridor carves and boosts the corridor to a proposed road's factor (10: a paved
m^2 counts ten). Stored beside each network's frames and in SCRATCH for the clearings, then
folded into the page's rows as `C` (m^2).

    S=research/roadless/atlas/roads5810/sheet.py       # from the repo root, PYTHONPATH=.
    uv run python $S network <name> <run> <data dir>   # its carved mesh in <data dir>/.meshes
    uv run python $S clearings                         # the five clearings on the plain mesh
    uv run python $S page                              # `C` into DATA and CDATA
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import shapely

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "research" / "roadless"))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402
import metrics  # noqa: E402
import render  # noqa: E402

BID = "ZAF.9.3.1_1_5810"
SITE = ROOT / ".superpowers/sdd/2026-10-22-site/roads5810"       # the lineup's two networks
FLOW = ROOT / ".superpowers/sdd/2026-10-22-flow-roads"          # the studies' networks
SCRATCH = ROOT / ".superpowers/sdd/2026-10-26-clearings5810"    # clear5810.SCRATCH
CLEARINGS = ("simp", "j2", "ff", "greedy", "cheap")             # clear5810's keys
# per page network its frames' runs (the lineup's page rows mix its two passes) and their folder
RUNS = {"cycle_native": (SITE, ("coarse", "fine")), "clearance_looped": (SITE, ("coarse", "fine"))}
TOL = 1e-9        # relative: a frame two runs share, the same carving prefix's sheet


def metric():  # type: ignore[no-untyped-def]
    return gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU())


def network(name: str, run: str, *, data: Path) -> None:
    """<name>_<run>'s frames' sheets, the carving rebuilt as compute.times_of rebuilds it. Writes
    <name>_<run>_sheet.json into `data`: the baseline's and per stop m its sheet."""
    t0 = time.time()
    d = np.load(data / f"{name}_{run}.npz", allow_pickle=True)          # our own file: WKB
    stops = [int(s) for s in d["stops"]]
    met = metric()
    [b] = common.build_blocks([BID])
    carving = gcl.Carving.of(gc.geometry(b),
                             [[shapely.from_wkb(d[f"corridor_{mm}"])[0] for mm in stops]])
    m = met.model(b, met.fixed(b, cache=data / ".meshes", carving=carving),
                  demand=common.PopulationDemand())
    out = dict(base=m.sheet(m.factors(np.zeros(m.n))),
               stops={str(mm): m.sheet(m.road_factors(gcl.Prefix(net=0, stop=k)))
                      for k, mm in enumerate(stops, start=1)})
    (data / f"{name}_{run}_sheet.json").write_text(json.dumps(out, indent=1))
    print(f"{name}_{run}: {len(stops)} stops' sheets in {time.time() - t0:.0f}s", flush=True)


def clearings() -> None:
    """Each clearing's frames' sheets on the plain mesh (clear5810's model). Writes
    SCRATCH/sheet.json: the baseline's and per method, per frame fff its sheet."""
    [b] = common.build_blocks([BID])
    m = metric().clearing(b, demand=common.PopulationDemand())
    out: dict = dict(base=m.sheet(m.factors(np.zeros(m.n))), methods={})  # type: ignore[type-arg]
    for key in CLEARINGS:
        d = np.load(SCRATCH / f"{key}.npz")
        out["methods"][key] = {k[2:]: m.sheet(m.factors(m.k_of(d[k].astype(bool))))
                               for k in d.files if k.startswith("x_")}
    (SCRATCH / "sheet.json").write_text(json.dumps(out, indent=1))
    print(f"{len(CLEARINGS)} clearings' sheets", flush=True)


def runs_of(method: str) -> tuple[Path, tuple[str, ...]]:
    """A page network's folder and runs: the lineup's two (RUNS), or the one run of the
    flow-roads study's whose frames are `method`'s."""
    if method in RUNS:
        return RUNS[method]
    [run] = [p.name[len(method) + 1:-len(".json")] for p in FLOW.glob(f"{method}_*.json")
             if re.fullmatch(r"slider|dp|greedy", p.name[len(method) + 1:-len(".json")])]
    return FLOW, (run,)


def added(method: str) -> dict[str, float]:
    """Per frame DDD of a page network the conductance its stop adds (m^2) over its mesh's own
    baseline; a frame two runs share agrees in both (the same prefix of the same corridors)."""
    data, runs = runs_of(method)
    out: dict[str, float] = {}
    for run in runs:
        s = json.loads((data / f"{method}_{run}_sheet.json").read_text())
        names = render.frame_names(json.loads((data / f"{method}_{run}.json").read_text()))
        for mm, DDD in names.items():
            c = s["stops"][str(mm)] - s["base"]
            if DDD in out and abs(out[DDD] - c) > TOL * s["base"]:
                raise AssertionError(f"{method} {DDD}: {out[DDD]} in one run, {c} in another")
            out[DDD] = c
    return out


def page() -> None:
    """`C` (m^2, rounded to 1) into each row of the page's DATA and CDATA, by its frame."""
    path = HERE / "index.html"
    html = path.read_text()
    cs = json.loads((SCRATCH / "sheet.json").read_text())
    for var in ("DATA", "CDATA"):
        rows = json.loads(re.search(rf"var {var} = (\{{.*?\}});\n", html)[1])  # type: ignore[index]
        for method, frames in rows.items():
            c = (added(method) if var == "DATA"
                 else {f: v - cs["base"] for f, v in cs["methods"][method].items()})
            for r in frames:
                r["C"] = round(c[r["f"]])
        text = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        html = re.sub(rf"var {var} = \{{.*?\}};\n", lambda _, v=var, t=text: f"var {v} = {t};\n",
                      html, count=1)
    path.write_text(html)
    print("C into the page's rows", flush=True)


if __name__ == "__main__":
    arg, *rest = sys.argv[1:]
    if arg == "network" and len(rest) == 3:
        network(rest[0], rest[1], data=Path(rest[2]))
    elif arg == "clearings" and not rest:
        clearings()
    elif arg == "page" and not rest:
        page()
    else:
        raise SystemExit(f"{sys.argv[1:]}: network <name> <run> <data dir> | clearings | page")
