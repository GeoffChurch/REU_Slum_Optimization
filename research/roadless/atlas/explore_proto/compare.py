"""Every method on the Demo's block, drawn as the Demo draws Shock Reblock (owner 2026-11-03: "a
method selector for the 5810 network visualization, and additional series in the metrics plots"):
each network of the 5810 build-out page (roads5810/index.html's DATA: the methods, their rows per
slider budget, their names and blurbs in NETS and KEY) as build.py's escape-time plate, one
sprite sheet per method (build.sheet_of), and the rows as JSON; for the comparison artifact
(artifact.py beside this file).

A method's frames are found as roads5810/sheet.py finds them (sheet.RUNS, else the one study
folder holding its runs: sheet.runs_of), each stop's plate drawn from the stored corridor, damage
and escape times (build.stored_plates' reading, its frame checked against the baseline's).

    C=research/roadless/atlas/explore_proto/compare.py     # from the repo root, PYTHONPATH=.
    uv run python $C plates <workers>       # every frame's plate into build.FRAMES/<block>/cmp/
    uv run python $C sheets                 # the sprite sheets and methods.json into OUT
    uv run python $C faceoff ~/src/wt_demo  # FACEOFF's grids on the Demo's 41785 slider: their
                                            # plates, sheets and series into OUT/d/faceoff.json
    uv run python $C artifact ~/src/wt_demo # compare.html and its files into OUT (the Demo's
                                            # 41785 bundle and the ScreenMap bundles from a site
                                            # checkout that `build.py site` has written)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Any, NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "roads5810"))
import build as bd  # noqa: E402  (puts research/roadless and roads5810 on the path)
import local  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
import sheet as sh  # noqa: E402
from block_site import Site  # noqa: E402

nw, render = bd.nw, bd.render
PAGE_HTML = HERE.parent / "roads5810" / "index.html"
CMP = bd.FRAMES / bd.PAGE.bid / "cmp"           # the plates: <class>_<DDD>.webp, 000 the baseline
OUT = bd.PLATES / "compare"                     # the sheets and methods.json
WIDTH = bd.WIDTH                                # px: a sheet frame's width (the Demo's)


class Stop(NamedTuple):
    """Where a frame's network is stored: its folder, the method's key there, its run, its stop."""
    data: Path
    key: str
    run: str
    mm: int


def page() -> dict[str, Any]:
    html = PAGE_HTML.read_text()
    return {v: local.js(html, v) for v in ("DATA", "NETS", "CLASS", "KEY")}


def stops_of(key: str, frames: list[str]) -> dict[str, Stop]:
    """Per frame DDD of the method `key` its stored stop."""
    data, runs = sh.RUNS[key] if key in sh.RUNS else sh.runs_of(key, studies=sh.STUDIES)
    out: dict[str, Stop] = {}
    for run in runs:
        info = json.loads((data / f"{key}_{run}.json").read_text())
        for mm, ddd in render.frame_names(info).items():
            out.setdefault(ddd, Stop(data, key, run, mm))
    missing = [f for f in frames if f not in out]
    if missing:
        raise SystemExit(f"{key}: frames {missing} not in its runs {runs} ({data})")
    return {f: out[f] for f in frames}


W: Any = None          # the block's World, built once in the parent and inherited by the workers


def draw(job: tuple[Stop, Path]) -> str:
    s, out = job
    d = np.load(s.data / f"{s.key}_{s.run}.npz", allow_pickle=True)            # our own: WKB
    t = np.load(s.data / f"{s.key}_{s.run}_times.npz")
    if not np.allclose(d["frame"], np.asarray(W.fr.frame, dtype=float)):
        raise AssertionError(f"{s.key} {s.run}: its frame is not the baseline's")
    [corridor] = shapely.from_wkb(d[f"corridor_{s.mm}"])
    bd.homes_plate(W, t[f"u_{s.mm}"], c=d[f"c_{s.mm}"], corridor=corridor, out=out)
    return out.name


def plates(workers: int) -> None:
    global W
    p = page()
    jobs: list[tuple[Stop, Path]] = []
    for key, rows in p["DATA"].items():
        cls = p["CLASS"][key]
        for f, s in stops_of(key, local.frames(rows)).items():
            out = CMP / f"{cls}_{f}.webp"
            if not out.is_file():
                jobs.append((s, out))
    CMP.mkdir(parents=True, exist_ok=True)
    site = Site.of(bd.PAGE.bid)
    W = bd.world(site, bd.context_source())
    base = CMP / "base_000.webp"
    if not base.is_file():
        bd.homes_plate(W, W.u_plain, c=np.zeros(len(W.polys)), corridor=None, out=base)
    print(f"{len(jobs)} plates to draw on {workers} workers", flush=True)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    with get_context("fork").Pool(workers) as pool:
        for i, name in enumerate(pool.imap_unordered(draw, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} {name}", flush=True)


def sheets() -> None:
    """Per method its sheet (the baseline first, then its frames in slider order) and its rows
    per budget (null where it has no network), into OUT/methods.json with the page's order,
    names and blurbs."""
    p = page()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s").mkdir(exist_ok=True)
    order = [k[1] for k in p["KEY"]]                 # the page's key order, by class
    by_cls = {c: k for k, c in p["CLASS"].items()}
    point = {k[1] for k in p["KEY"] if len(k) > 2 and k[2] == "point"}
    label = {k[1]: k[0] for k in p["KEY"]}
    methods = []
    for cls in order:
        key = by_cls[cls]
        rows = p["DATA"][key]
        fs = ["000", *local.frames(rows)]
        paths = [CMP / "base_000.webp", *[CMP / f"{cls}_{f}.webp" for f in fs[1:]]]
        sheet = bd.sheet_of(paths, OUT / "s" / f"{cls}.webp", WIDTH)
        name, blurb = p["NETS"][key]
        methods.append(dict(
            cls=cls, key=key, label=label[cls], name=name, blurb=blurb,
            per_budget=cls in point, sheet=bd.dataclasses.asdict(sheet),
            rows=[None if r is None else dict(D=r["D"], km=r["km"], LP=r["LP"], n=r.get("n"),
                                              C=r.get("C"), row=fs.index(r["f"]))
                  for r in rows]))
        print(f"  {cls}: {len(fs)} frames", flush=True)
    site = Site.of(bd.PAGE.bid)
    (OUT / "methods.json").write_text(json.dumps(dict(
        block_id=bd.PAGE.bid, budgets=list(nw.SLIDER), escape_scale=bd.escape_scale(site),
        methods=methods), ensure_ascii=False))
    mb = sum(f.stat().st_size for f in (OUT / "s").iterdir()) / 1e6
    print(f"{len(methods)} methods, sheets {mb:.1f} MB", flush=True)


# The face-off's further methods (owner 2026-11-03: the spur runs beside Shock Reblock's loops):
# each a grid on 41785 (atlas/buildout.py `table`), with the label the picker shows.
FACEOFF = (
    (bd.Grid("ears.paths.b20.f100.e30.spur10-30.nodt.gl0.25Croad.n2s0.1.M8.K4.D0.91.v3",
             stem="es30", stops_because="no road left that would split a sub-block or end a spur"),
     "Shock Reblock with spurs of 10 to 30 m"),
    (bd.Grid("ears.paths.b20.f100.e30.spur10-80.nodt.gl0.25Croad.n2s0.1.M8.K4.D0.91.v3",
             stem="es80", stops_because="no road left that would split a sub-block or end a spur"),
     "Shock Reblock with spurs of 10 to 80 m"),
)


def faceoff(root: Path) -> None:
    """FACEOFF's grids on the Demo's 41785 build-out (its budgets, read from the site checkout
    `root`): each grid's plates, its sheet and its series, into OUT/d/faceoff.json."""
    demo = root / bd.SITE_DIR
    bundle = json.loads((demo / "demo.json").read_text())
    b = json.loads((demo / bundle["comparison"]["buildout"]).read_text())
    site = Site.of(b["block_id"])
    scale = bd.escape_scale(site)
    (OUT / "d").mkdir(parents=True, exist_ok=True)
    w = bd.world(site, bd.context_source())
    series = []
    for g, label in FACEOFF:
        g.plates(site, w)
        t = g.table(site)
        series.append(bd.grid_series(site, scale, g, tuple(b["budgets"]), method=g.key,
                                     label=label,
                                     run=bd.DemoRecorded(kind="recorded",
                                                         seconds=t["whole"]["seconds"],
                                                         machine=bd.METHOD.runtime.machine),
                                     out=OUT / "d"))
        print(f"  {g.stem}: {len(series[-1].cells)} budgets", flush=True)
    (OUT / "d" / "faceoff.json").write_text(json.dumps(
        [bd.dataclasses.asdict(x) for x in series], ensure_ascii=False))


def artifact(root: Path) -> None:
    """compare.html as OUT/index.html beside the sheets, with the Demo's 41785 build-out and its
    sheets (OUT/d/) and the ScreenMap bundles (OUT/screen/) copied from the site checkout `root`."""
    demo, screen = root / bd.SITE_DIR, root / "examples" / "screen-map"
    if not (demo / "demo.json").is_file():
        raise SystemExit(f"{root}: no {bd.SITE_DIR}/demo.json (run build.py site {root})")
    if not (OUT / "methods.json").is_file():
        raise SystemExit(f"{OUT}: no methods.json (run `sheets`)")
    bundle = json.loads((demo / "demo.json").read_text())
    name = bundle["comparison"]["buildout"]
    b = json.loads((demo / name).read_text())
    (OUT / "d").mkdir(exist_ok=True)
    (OUT / "screen").mkdir(exist_ok=True)
    for f in [name, *[s["sheet"]["file"] for s in b["series"]]]:
        shutil.copy2(demo / f, OUT / "d" / f)
    for city in ("capetown", "nairobi"):
        shutil.copy2(screen / f"{city}.json", OUT / "screen" / f"{city}.json")
    shutil.copy2(HERE / "compare.html", OUT / "index.html")
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(f"{OUT}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB",
          flush=True)


def main(argv: list[str]) -> None:
    match argv:
        case ["plates", n]:
            plates(int(n))
        case ["sheets"]:
            sheets()
        case ["faceoff", root]:
            faceoff(Path(root).expanduser())
        case ["artifact", root]:
            artifact(Path(root).expanduser())
        case _:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
