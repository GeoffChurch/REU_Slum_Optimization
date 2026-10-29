"""The conductance each frame of the 5810 build-out page adds (owner 2026-10-27: the roads' and the
clearings' costs in one unit, a monotone function of the conductance added, the identity, since a
footprint's resistance may be infinite): per frame the sheet conductance its state holds
(gap_clear.Model.sheet: per triangle area x factor x boost, in m^2 of open ground) less its own
mesh's baseline's. A clearing opens its footprints and the shy zones they alone held shut; a road
opens what its corridor carves and boosts the corridor to a proposed road's factor (10: a paved
m^2 counts ten). Stored beside each network's frames and in SCRATCH for the clearings, then
folded into the page's rows as `C` (m^2). A network's sheets are written by the pass that draws
its frames (compute.frames_of: <name>_<run>_sheet.json beside them).

    S=research/roadless/atlas/roads5810/sheet.py       # from the repo root, PYTHONPATH=.
    uv run python $S clearings                         # the five clearings on the plain mesh
    uv run python $S page                              # `C` into DATA and CDATA
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "research" / "roadless"))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_factor  # noqa: E402
import metrics  # noqa: E402
import render  # noqa: E402

cp = render.cp
PAGE = cp.PAGE                    # the page's block: its lineup's and networks.py's frames
FLOW = ROOT / ".superpowers/sdd/2026-10-22-flow-roads"          # the flow-roads study's networks
STUDIES = (FLOW, PAGE.networks)   # the page's study folders (networks.py's: PAGE.networks)
STUDY_RUN = r"slider|dp|greedy|at[0-9]{3}"     # a study network's runs (networks.py's: at<DDD>)
SCRATCH = ROOT / ".superpowers/sdd/2026-10-26-clearings5810"    # clear5810.SCRATCH
CLEARINGS = ("simp", "j2", "ff", "greedy", "cheap")             # clear5810's keys
# per page network its frames' runs (the lineup's page rows mix its two passes) and their folder
RUNS = {"cycle_native": (PAGE.data, ("coarse", "fine")),
        "clearance_looped": (PAGE.data, ("coarse", "fine"))}
TOL = 1e-9        # relative: a frame two runs share, the same carving prefix's sheet


def metric():  # type: ignore[no-untyped-def]
    return gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU())


def clearings() -> None:
    """Each clearing's frames' sheets on the plain mesh (clear5810's model). Writes
    SCRATCH/sheet.json: the baseline's and per method, per frame fff its sheet."""
    [b] = common.build_blocks([PAGE.bid])
    m = metric().clearing(b, demand=common.PopulationDemand())
    out: dict = dict(base=m.sheet(m.factors(np.zeros(m.n))), methods={})  # type: ignore[type-arg]
    for key in CLEARINGS:
        d = np.load(SCRATCH / f"{key}.npz")
        out["methods"][key] = {k[2:]: m.sheet(m.factors(m.k_of(d[k].astype(bool))))
                               for k in d.files if k.startswith("x_")}
    (SCRATCH / "sheet.json").write_text(json.dumps(out, indent=1))
    print(f"{len(CLEARINGS)} clearings' sheets", flush=True)


def runs_of(method: str, *, studies: tuple[Path, ...]) -> tuple[Path, tuple[str, ...]]:
    """A network's folder and runs: the runs (STUDY_RUN) of the one folder of `studies` whose
    frames are `method`'s."""
    found = [(d, tuple(sorted(r))) for d in studies
             if (r := [p.name[len(method) + 1:-len(".json")] for p in d.glob(f"{method}_*.json")
                       if re.fullmatch(STUDY_RUN, p.name[len(method) + 1:-len(".json")])])]
    [(data, runs)] = found
    return data, runs


def added(method: str, *, data: Path, runs: tuple[str, ...]) -> dict[str, float]:
    """Per frame DDD of a network (its `runs` in `data`) the conductance its stop adds (m^2) over
    its mesh's own baseline; a frame two runs share agrees in both (the same prefix of the same
    corridors)."""
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
            if var == "DATA":
                data, runs = (RUNS[method] if method in RUNS
                              else runs_of(method, studies=STUDIES))
                c = added(method, data=data, runs=runs)
            else:
                c = {f: v - cs["base"] for f, v in cs["methods"][method].items()}
            for r in frames:
                if r is not None:          # a per-budget method's slider step below its first
                    r["C"] = round(c[r["f"]])
        text = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        html = re.sub(rf"var {var} = \{{.*?\}};\n", lambda _, v=var, t=text: f"var {v} = {t};\n",
                      html, count=1)
    path.write_text(html)
    print("C into the page's rows", flush=True)


if __name__ == "__main__":
    arg, *rest = sys.argv[1:]
    if arg == "clearings" and not rest:
        clearings()
    elif arg == "page" and not rest:
        page()
    else:
        raise SystemExit(f"{sys.argv[1:]}: clearings | page")
