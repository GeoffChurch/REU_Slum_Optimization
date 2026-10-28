"""The 5810 build-out page's networks from the pilots of 2026-10-27 (owner 2026-10-28): roads grown
from the street as a front (research/roadless/growth.py), reblocking by ears (ears.py), the joint
SIMP's answers (road_simp.py rev 8, `.cjoint`), the partition SIMP's (partition.py) and the grid
templates' (templates.py). Each is read from its study's rows into a compute.Network and built
out like the page's other networks: its frames' rasters, lenses, escape times and sheets
(compute.frames_of: one heavy process, ~9 GB a carved mesh on 5810), its plates (render.py's),
its page rows (DATA: f, D, n, km, LP, C) and entries (NETS, CLASS, DOT, KEY, POINTS), then the
artifact (artifact.py).

A family (Family, a Strategy, chosen on the command line) names the method, styles it and builds
its source (Source, a Strategy):
  Grown: a growth or ears run's steps (growth_rows/<block>/<plan>/: tri_step.parquet, per
    triangle of the eps world's mesh, the plain mesh, the step it was grown in; steps.parquet,
    a row per state). Prefix s is the triangles grown before step s, unioned and smoothed as the
    growth study's frames are (.superpowers/sdd/2026-10-27-growth/frames.py's `region`:
    road_simp.SMOOTH_M); the shortest prefix whose exact D reaches each slider target is its
    frame (the whole run where it falls short). Two smoothings part by a few m^2 (4 -- 10 m^2
    between neighbouring steps on 5810, over Carving.of's tolerance), so each stop's corridor is
    unioned with the last stop's, on the raw unions' GRID_M grid (off it, two stops' outlines
    left vertices 1e-9 m apart, which the carved mesh's conformity refused on the islands run):
    the stops nest, a stop's D read on what is carved (its own region's, which picked it, within
    the grid's half millimetre). Carved as a kappa-10 road under RUNTIME whatever the
    run's contrast: the grounded runs too, so every method on the page compares under one
    metric. A line on the chart.
  Answers: an optimizer's answer per budget (one run each, `at<DDD>`, DDD its budget): its paved
    ground, its roads as lines where it has them, the buildings it clears beside them (the joint
    SIMP's: proj(y) >= 1/2 at the last beta); one frame, its own mesh (road_simp.scored's
    carving, so road_simp.MESHES caches it), its L_P and D checked against the optimizer's own
    score. Diamonds on the chart; the plates hold the answer below the slider's budget, the
    baseline below the first. Its runs (Runs, resolved once from the spec): the rows' folders
    whose names the spec matches, {D} each one's budget and any other field ({n}) a number bound
    per budget by the one run there (the partition SIMP's start, its size changing with D); the
    method's key the spec without D{D}., each other field its values joined (spec20,40,64s0).
n is a frame's road pieces (its lines; else its ground's skeleton on a 1 m raster, merged between
junctions, as road_simp's roads answer is read), km their length. C is sheet.added's (the stop's
sheet less its own mesh's baseline).

    S=research/roadless/atlas/roads5810/networks.py    # from the repo root, PYTHONPATH=.
    uv run python $S add <family> <spec..>             # frames, plates, page, artifact.py
    uv run python $S frames|plates|page <family> <spec..>
    uv run python $S ready                             # finished runs not yet on the page

families and specs:
  growth <plan>                               growth_rows/<block>/<plan> (growth.Plan's name)
  ears <plan>                                 the same folder (ears.Plan's name)
  joint paved|roads <plan with D{D}>          road_simp_rows/<RUNTIME>/<plan> at each D it holds
  partition band|cut <plan with D{D}[, {n}]>  partition_rows/<RUNTIME>/<plan>, the same ({n}: the
                                              start's size, one run a budget)
  templates <plan with D{D}>                  templates_rows/<RUNTIME>/<plan>, the same

On the page (2026-10-28), each one `add`:
  growth gr.k10.att.greedy.N48.sep14.n2s0.1.B21m8k2.K0.D0.15.v2
  growth gr.k10.att.e1s0.N48.sep14.n2s0.1.B21m8k2.K0.D0.15.v2
  growth gr.gl0.25Croad.att.greedy.N48.sep14.n2s0.1.B0.K0.D0.15.v2
  growth gr.gl0.25Croad.att.e1s0.N48.sep14.n2s0.1.B0.K0.D0.15.v2
  joint paved 'rs.D{D}.s0.1.r7.q1,2,3,3,3.b1,2,4,8,16.i60.uOC.cjoint.p1e0.0001v8'
  joint roads 'rs.D{D}.s0.1.r7.q1,2,3,3,3.b1,2,4,8,16.i60.uOC.cjoint.p1e0.0001v8'
  growth gr.k10.att.greedy.N48.sep14.n2s0.3.B21m8k2.K0.D0.15.v2           (price 0.3)
  ears ears.paths.b20.gl0.25Croad.n2s0.1.M8.K4.D0.15.v1
  templates 'tp.D{D}.s0.1.v1'
  growth gr.gl0.25Croad.free0.03.greedy.N48.sep14.n2s0.1.B0.K4.D0.15.v2  (islands: the charge
      that scores best under RUNTIME, the growth study's frames.py score: 0.492 at D 0.15, 0.01's
      0.458, 0.003's 0.465)
  ears ears.divides.p0.1.gl0.25Croad.n2s0.1.M16.K4.D0.15.v2         (rev 2: rev 1 starved the
      large faces, ears.py 8794c3d7)
  partition cut 'pt.K5.R10.t2,1.5,1.2,1,1.b1,2,4,8,16.q1,2,3,3,3.i40.D{D}.s0.1.spec{n}s0.p1e0.0001v2'
  partition band (the same spec; the spectral start's 20, 40 and 64 clusters at D 0.05, 0.10, 0.15)
`ready` lists the finished runs not on the page, each once, with the spec naming the most budgets
(D{D} alone, or with {n}); each one `add`, its style in place.
A combination with no style here (STYLES, the families' `entry`) is refused: give it a class in
index.html and a row here.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import artifact  # noqa: E402
import compute as cp  # noqa: E402
import growth as gr  # noqa: E402
import local  # noqa: E402
import metrics  # noqa: E402
import relax  # noqa: E402
import render  # noqa: E402
import road_simp as rs  # noqa: E402
import sheet  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402
from reblock.destroyed import IncrementalDestroyed  # noqa: E402
from reblock.methods.betweenness.thinning import skeletonize  # noqa: E402

common, gc, gcl = cp.common, cp.gc, cp.gcl
BID = cp.BID
DATA = sheet.NETWORKS                 # the frames' rasters, jsons, times, sheets and meshes
SLIDER = render.SLIDER                # D 0.01 .. 0.15
SKEL_PIX = 1.0                        # m: the skeleton's raster (road_simp.PIXEL's)
GRID_M = 1e-3                         # m: a grown region's union grid (frames.py's `region`)


# ------------------------------------------------------------------ geometry

def skeleton(ground_local: Any) -> tuple[int, float]:
    """(pieces, metres) of a paved ground's skeleton: a SKEL_PIX raster of it thinned (Zhang-Suen),
    the pixels' 8-neighbour links merged between junctions (road_simp.roads' reading)."""
    x0, y0, x1, y1 = ground_local.bounds
    lo = np.array([x0, y0]) - 2 * SKEL_PIX
    shape = (int(np.ceil((x1 - lo[0]) / SKEL_PIX)) + 3, int(np.ceil((y1 - lo[1]) / SKEL_PIX)) + 3)
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    shapely.prepare(ground_local)
    on = shapely.contains_xy(ground_local, lo[0] + (i.ravel() + 0.5) * SKEL_PIX,
                             lo[1] + (j.ravel() + 0.5) * SKEL_PIX).reshape(shape)
    skel = skeletonize(on)
    ii, jj = np.nonzero(skel)
    keep = set(zip(ii.tolist(), jj.tolist(), strict=True))
    segs = [((a + 0.5, b + 0.5), (a + da + 0.5, b + db + 0.5)) for a, b in keep
            for da, db in ((0, 1), (1, -1), (1, 0), (1, 1)) if (a + da, b + db) in keep]
    if not segs:
        return 0, 0.0
    xy = lo + SKEL_PIX * np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)
    lines = shapely.get_parts(shapely.line_merge(shapely.multilinestrings(shapely.linestrings(xy))))
    return len(lines), float(shapely.length(lines).sum())


def smoothed(g: Any) -> Any:
    """A union of triangles closed, opened and simplified by road_simp.SMOOTH_M (road_simp.paved_of,
    frames.py's `region`)."""
    g = shapely.buffer(shapely.buffer(g, rs.SMOOTH_M, quad_segs=1), -rs.SMOOTH_M, quad_segs=1)
    return shapely.make_valid(shapely.simplify(g, rs.SMOOTH_M))


def polygonal(g: Any) -> Any:
    """`g`'s polygons (make_valid can leave a collapsed sliver as a line, which an overlay on a
    grid refuses as mixed-dimension input)."""
    parts = shapely.get_parts(g)
    return shapely.union_all(parts[np.isin(shapely.get_type_id(parts), (3, 6))])


def area_weights(b) -> NDArray[np.float64]:  # type: ignore[no-untyped-def]
    return np.asarray(common.POPULATIONS["area"].weights(np.asarray(b.buildings.outlines)))


# ------------------------------------------------------------------ the networks

class Grown:
    """A growth (or ears) run's steps as a compute.Network (module docstring): prefixes 1 ..
    `size`, the stops those the slider's `targets` pick, and the whole run."""

    def __init__(self, b, run: Path, *, targets: tuple[float, ...]) -> None:  # type: ignore[no-untyped-def]
        t0 = time.time()
        self.b = b
        self.geo = gc.geometry(b)
        metric = gcl.complex_of(metrics.RUNTIME, factorization=cp.gap_factor.SuperLU())
        fm = metric.fixed(b, cache=gcl.CACHE, carving=gcl.Carving.none(self.geo)).fm
        step = pd.read_parquet(run / "tri_step.parquet").step.to_numpy()
        rows = pd.read_parquet(run / "steps.parquet")
        if len(step) != fm.nt:
            raise ValueError(f"{run.name}: {len(step)} triangles' steps on a mesh of {fm.nt}")
        self.size = int(rows.step.max())
        if step.max() >= self.size:
            raise ValueError(f"{run.name}: a triangle grown at step {step.max()}, the rows end "
                             f"at state {self.size}")
        self.W = area_weights(b)
        # the raw unions in step order (each the last's and its step's triangles)
        self.raw: list[Any] = [shapely.Polygon()]
        for s in range(self.size):
            t = np.flatnonzero(step == s)
            self.raw.append(shapely.union_all(
                np.concatenate([[self.raw[-1]], shapely.polygons(fm.xy[fm.tri[t]])]),
                grid_size=GRID_M))
        self._D: dict[int, float] = {}
        D_eps = rows.set_index("step").D
        stops = {t: self._shortest(t, guess=int(D_eps.index[np.argmax(D_eps.to_numpy() >= t)])
                                   if (D_eps >= t).any() else self.size) for t in targets}
        self.stop_of = {t: s for t, s in stops.items() if self.D_region(s) >= t}
        chain = sorted(set(self.stop_of.values()) | {self.size})
        self.corridors: dict[int, Any] = {}
        last = None
        for s in chain:
            reg = shapely.set_precision(polygonal(self.region(s)), GRID_M)
            last = reg if last is None else polygonal(shapely.make_valid(
                shapely.union(last, reg, grid_size=GRID_M)))
            self.corridors[s] = last
        print(f"{run.name}: {self.size} steps, stops {self.stop_of} ({time.time() - t0:.0f}s)",
              flush=True)

    def region(self, s: int) -> Any:
        """Prefix s's region (local frame): its triangles' union smoothed."""
        return smoothed(self.raw[s])

    def D_region(self, s: int) -> float:
        """D (area weights) of prefix s's own region (not chained), cached."""
        if s not in self._D:
            c = common.MEASURE.c(self.b.buildings, self.geo.utm(self.region(s)))
            self._D[s] = float((np.asarray(c) * self.W).sum() / self.W.sum())
        return self._D[s]

    def _shortest(self, target: float, *, guess: int) -> int:
        """The shortest prefix whose region's D reaches `target` (D grows with the prefix), from
        the eps world's guess down or up."""
        s = guess
        if self.D_region(s) >= target:
            while s > 1 and self.D_region(s - 1) >= target:
                s -= 1
            return s
        while s < self.size and self.D_region(s) < target:
            s += 1
        return s

    def prefix_len(self, target: float) -> int:
        return self.stop_of[target] if target in self.stop_of else self.size

    def corridor(self, mm: int) -> Any:
        return self.geo.utm(self.corridors[mm])

    def cleared(self, mm: int) -> NDArray[np.bool_]:
        return np.zeros(len(self.W), dtype=bool)

    def c(self, mm: int) -> NDArray[np.float64]:
        return np.asarray(common.MEASURE.c(self.b.buildings, self.corridor(mm)), dtype=float)

    def roads(self, mm: int) -> tuple[int, float]:
        return skeleton(self.corridors[mm])


class Answer(NamedTuple):
    """An optimizer's answer at one budget: its paved ground (world frame), its roads as lines
    (None: the ground's skeleton), the buildings it clears beside them, its own L_P and D (the
    check)."""
    budget: float
    ground: Any
    lines: Any
    cleared: NDArray[np.bool_]
    L_P: float
    D: float


class One:
    """One answer as a compute.Network of one prefix."""
    size = 1

    def __init__(self, b, a: Answer) -> None:  # type: ignore[no-untyped-def]
        self.b, self.a = b, a
        self.geo = gc.geometry(b)
        inc = IncrementalDestroyed(common.MEASURE, b.buildings)      # road_simp._destroyed's
        inc.add(a.ground)
        if a.cleared.any():
            inc.add(shapely.union_all(np.asarray(b.buildings.outlines)[a.cleared]))
        self._c = np.asarray(inc.c(), dtype=float)
        W = area_weights(b)
        D = float((self._c * W).sum() / W.sum())
        if abs(D - a.D) > 1e-9:
            raise AssertionError(f"D {D} against the answer's own {a.D}")

    def prefix_len(self, target: float) -> int:
        return 1

    def corridor(self, mm: int) -> Any:
        return self.a.ground

    def cleared(self, mm: int) -> NDArray[np.bool_]:
        return self.a.cleared

    def c(self, mm: int) -> NDArray[np.float64]:
        return self._c

    def roads(self, mm: int) -> tuple[int, float]:
        if self.a.lines is None:
            return skeleton(self.geo.local(self.a.ground))
        return len(self.a.lines), float(shapely.length(self.a.lines).sum())


# ------------------------------------------------------------------ sources

class Run(NamedTuple):
    """A source's run: its name (<method>_<run> in DATA) and how to build its Network."""
    name: str
    budget: float | None          # an answer's budget; None for a grown run's slider


class Source(Protocol):
    """Where a page method's frames come from (module docstring)."""

    @property
    def points(self) -> bool: ...

    def ready(self) -> bool: ...

    def runs(self) -> list[Run]: ...

    def frames(self, key: str, b) -> None:  # type: ignore[no-untyped-def]
        """compute.frames_of for each run, into DATA."""
        ...

    def rows(self, key: str) -> list[dict[str, Any] | None]:
        """The page's fifteen slider rows (render.SLIDER), C included."""
        ...


def page_row(r: dict[str, Any], f: str, C: dict[str, float]) -> dict[str, Any]:
    """A frames_of row as the page's: f, D, n, km, LP, C."""
    return dict(f=f, D=round(r["D_area"], 4), n=r["n_roads"], km=round(r["road_m"] / 1000, 2),
                LP=round(r["L_P"], 3), C=round(C[f]))


def info_of(key: str, run: str) -> dict[str, Any]:
    return json.loads((DATA / f"{key}_{run}.json").read_text())


class GrownSource:
    """A growth or ears run's steps (Grown), one run `slider`."""
    points = False

    def __init__(self, plan: str, *, budget: float) -> None:
        self.dir = gr.OUT / BID / plan
        self.budget = budget

    def ready(self) -> bool:
        f = self.dir / "steps.parquet"
        return f.exists() and float(pd.read_parquet(f).D.iloc[-1]) >= self.budget

    def runs(self) -> list[Run]:
        return [Run(name="slider", budget=None)]

    def frames(self, key: str, b) -> None:  # type: ignore[no-untyped-def]
        t0 = time.time()
        net = Grown(b, self.dir, targets=SLIDER)
        full = max(SLIDER) not in net.stop_of
        cp.frames_of(key, "slider", b, net, t0, frames=cp.Pass(targets=SLIDER, full=full),
                     data=DATA, meshes=DATA / ".meshes")

    def rows(self, key: str) -> list[dict[str, Any] | None]:
        info = info_of(key, "slider")
        names = render.frame_names(info)
        C = sheet.added(key)
        by_target = {round(t, 2): r for r in info["frames"] for t in r["targets"] if t is not None}
        [whole] = [r for r in info["frames"] if None in r["targets"]] or [None]
        out: list[dict[str, Any] | None] = []
        for t in SLIDER:
            if t in by_target:
                out.append(page_row(by_target[t], names[by_target[t]["m"]], C))
            elif whole is not None:
                out.append(dict(page_row(whole, names[whole["m"]], C), full=True))
            else:
                raise AssertionError(f"{key}: no frame at {t} and no whole run")
        return out


class Design(Protocol):
    """An optimizer's answers per budget (AnswerSource's)."""

    def budgets(self) -> list[float]: ...

    def ready(self, budget: float) -> bool: ...

    def answer(self, b, budget: float) -> Answer: ...  # type: ignore[no-untyped-def]

    def meshes(self) -> Path: ...


FIELD = re.compile(r"\{([A-Za-z]+)\}")       # a spec's field: {D}, {n}, ...


class Runs:
    """A spec's runs (module docstring), resolved once: the folders of `rows` that hold 5810's rows
    and whose names the spec's template matches, each field a number ({D} the run's budget, any
    other bound per budget by the one run there: a start's size, which changes with D), at most
    one run a budget. `key` is the template without its `D{D}.`, each other field its values in
    budget order joined by commas (spec{n}s0 -> spec20,40,64s0)."""

    def __init__(self, *, rows: Path, template: str) -> None:
        names = FIELD.findall(template)
        if "D" not in names or len(set(names)) != len(names):
            raise SystemExit(f"{template}: a spec holds D{{D}}, each field once")
        parts = FIELD.split(template)              # literal, field, literal, ...
        pat = re.compile("".join(re.escape(t) if i % 2 == 0 else rf"(?P<{t}>[0-9.]+)"
                                 for i, t in enumerate(parts)))
        found: dict[float, re.Match[str]] = {}
        for d in sorted(rows.iterdir()):
            if (m := pat.fullmatch(d.name)) and (d / f"{BID}.parquet").exists():
                D = float(m["D"])
                if D in found:
                    raise SystemExit(f"{template}: two runs at D {D:g}, {found[D].string} and "
                                     f"{d.name}")
                found[D] = m
        if not found:
            raise SystemExit(f"{template}: no run in {rows}")
        ms = [found[D] for D in sorted(found)]
        self.dirs: dict[float, Path] = {D: rows / found[D].string for D in sorted(found)}
        self.key = "".join(t if i % 2 == 0 else ",".join(m[t] for m in ms)
                           for i, t in enumerate(FIELD.split(template.replace("D{D}.", ""))))


def plan_names(runs: Runs) -> list[str]:
    """The runs' plans: their folders' names without the code's `.p<pixel>e<eps>v<rev>`."""
    return [re.sub(r"\.p[0-9.e-]+v[0-9]+$", "", d.name) for d in runs.dirs.values()]


def last_row(path: Path) -> pd.Series:
    return pd.read_parquet(path).iloc[-1]


class Joint:
    """The joint SIMP's answer `kind` (paved: its paved ground; roads: its linked roads, 7 m)
    beside its clearing (road_simp.answer's)."""

    def __init__(self, *, kind: str, runs: Runs) -> None:
        if kind not in ("paved", "roads"):
            raise SystemExit(f"{kind}: a joint answer is paved or roads")
        self.kind, self.runs = kind, runs

    def budgets(self) -> list[float]:
        return list(self.runs.dirs)

    def dir(self, budget: float) -> Path:
        return self.runs.dirs[budget]

    def ready(self, budget: float) -> bool:
        f = self.dir(budget) / f"{BID}.parquet"
        return f.exists() and np.isfinite(last_row(f)[f"{self.kind}_L_P"])

    def answer(self, b, budget: float) -> Answer:  # type: ignore[no-untyped-def]
        d = self.dir(budget)
        spec = re.sub(r"\.p[0-9.e-]+v[0-9]+$", "", d.name)
        beta = rs.Plan.of(spec).stages[-1][1]
        y = pd.read_parquet(d / f"{BID}_y.parquet")["y"].to_numpy()
        cleared = relax.project(y, beta) >= rs.HALF          # Roads.answer_clearing's, joint
        last = last_row(d / f"{BID}.parquet")
        if self.kind == "paved":
            ground, lines = gpd.read_parquet(d / f"{BID}_ground.parquet").geometry.iloc[0], None
        else:
            net = gpd.read_parquet(d / f"{BID}_roads.parquet")
            ground, lines = road_corridor(net), net.geometry.to_numpy()
        return Answer(budget=budget, ground=ground, lines=lines, cleared=cleared,
                      L_P=float(last[f"{self.kind}_L_P"]), D=float(last[f"{self.kind}_D_exact"]))

    def meshes(self) -> Path:
        return rs.MESHES


class Lined:
    """A partition or template answer: <block>[_<answer>]_ground.parquet and _lines.parquet, its
    score in the row's `<answer>_` columns; no clearing."""

    def __init__(self, *, runs: Runs, answer: str) -> None:
        self.runs, self.ans = runs, answer

    def budgets(self) -> list[float]:
        return list(self.runs.dirs)

    def dir(self, budget: float) -> Path:
        return self.runs.dirs[budget]

    def stem(self) -> str:
        return BID if self.ans == "best" else f"{BID}_{self.ans}"

    def ready(self, budget: float) -> bool:
        f = self.dir(budget) / f"{BID}.parquet"
        return f.exists() and np.isfinite(last_row(f)[f"{self.ans}_L_P"])

    def answer(self, b, budget: float) -> Answer:  # type: ignore[no-untyped-def]
        d = self.dir(budget)
        last = last_row(d / f"{BID}.parquet")
        ground = gpd.read_parquet(d / f"{self.stem()}_ground.parquet").geometry.iloc[0]
        lines = gpd.read_parquet(d / f"{self.stem()}_lines.parquet").geometry.to_numpy()
        return Answer(budget=budget, ground=ground, lines=lines,
                      cleared=np.zeros(len(b.buildings.outlines), dtype=bool),
                      L_P=float(last[f"{self.ans}_L_P"]), D=float(last[f"{self.ans}_D_exact"]))

    def meshes(self) -> Path:
        return rs.MESHES


def run_name(budget: float) -> str:
    return f"at{render.tag(budget)}"


class AnswerSource:
    """A Design's answers, a run each (`at<DDD>`), drawn as points."""
    points = True

    def __init__(self, design: Design) -> None:
        self.design = design

    def ready(self) -> bool:
        bs = self.design.budgets()
        return bool(bs) and all(self.design.ready(x) for x in bs)

    def runs(self) -> list[Run]:
        return [Run(name=run_name(x), budget=x) for x in self.design.budgets()]

    def frames(self, key: str, b) -> None:  # type: ignore[no-untyped-def]
        for run in self.runs():
            assert run.budget is not None
            t0 = time.time()
            a = self.design.answer(b, run.budget)
            cp.frames_of(key, run.name, b, One(b, a), t0, frames=cp.Pass(targets=(), full=True),
                         data=DATA, meshes=self.design.meshes())
            [r] = info_of(key, run.name)["frames"]
            if abs(r["L_P"] - a.L_P) > 1e-6:
                raise AssertionError(f"{key} {run.name}: L_P {r['L_P']} against the answer's "
                                     f"own {a.L_P}")
            print(f"{key} {run.name}: L_P {r['L_P']:.6f} (its own {a.L_P:.6f})", flush=True)

    def rows(self, key: str) -> list[dict[str, Any] | None]:
        C = sheet.added(key)
        got: dict[int, dict[str, Any]] = {}
        for run in self.runs():
            assert run.budget is not None
            info = info_of(key, run.name)
            [r] = info["frames"]
            got[round(run.budget * 100)] = dict(page_row(r, render.frame_names(info)[r["m"]], C),
                                                b=run.budget)
        out: list[dict[str, Any] | None] = []
        for k in range(1, len(SLIDER) + 1):
            below = [j for j in got if j <= k]
            if not below:
                out.append(None)
            elif max(below) == k:
                out.append(got[k])
            else:
                out.append(dict(got[max(below)], held=True))
        return out


# ------------------------------------------------------------------ families

class Entry(NamedTuple):
    """A method on the page: its DATA key (and plates' stem), dropdown name, how it works (one
    sentence, NETS), chart class (CLASS: unique, the artifact's sprite sheets are named by it),
    dot class (DOT, a line's), key label (KEY)."""
    key: str
    name: str
    how: str
    cls: str
    dot: str
    label: str


class Family(Protocol):
    def entry(self) -> Entry: ...

    def source(self) -> Source: ...


def price_words(price: float) -> str:
    return f"each 100,000 m² of conductance added priced at {price:g} of the baseline's P"


# (contrast grounded, free, greedy, price) -> (chart class, dot class): index.html's .gk ... .gli
STYLES = {(False, False, True, 0.1): ("gk", "grd"), (False, False, False, 0.1): ("gke", "grd"),
          (False, False, True, 0.3): ("gk3", "grd"), (True, False, True, 0.1): ("gl", "lkd"),
          (True, False, False, 0.1): ("gle", "lkd"), (True, True, True, 0.1): ("gli", "lkd")}


class Growth:
    """A growth.py run."""

    def __init__(self, plan: str) -> None:
        self.plan = plan
        self.p = gr.Plan.of(re.sub(r"\.v[0-9]+$", "", plan))
        if self.p.name != plan:
            raise SystemExit(f"{plan}: growth.Plan names it {self.p.name}")

    def entry(self) -> Entry:
        p = self.p
        grounded = p.contrast.grounded
        free = isinstance(p.connectivity, gr.Free)
        greedy = isinstance(p.pick, gr.Greedy)
        style = STYLES.get((grounded, free, greedy, p.cost.price))
        if style is None:
            raise SystemExit(f"{self.plan}: no page style (networks.STYLES)")
        world = "grounded" if grounded else "10×"
        pick = "greedy" if greedy else f"η {p.pick.eta:g}"          # type: ignore[attr-defined]
        name = f"growth, {world}, {pick}" + (", islands" if free else "") + (
            f", price {p.cost.price:g}" if p.cost.price != 0.1 else "")
        how = (f"Grows road from the street as a front on the fine mesh, {p.batch} road-width "
               f"patches a step at least {p.sep:g} m apart, "
               + ("the best first" if greedy else
                  f"each drawn with probability as its score to the power η {p.pick.eta:g}")  # type: ignore[attr-defined]
               + ", scored by the drop in P one solve predicts per cost (displacement, and "
               f"{price_words(p.cost.price)})"
               + (f", plus straight bridges up to {p.bridges.reach:g} m between roads where "  # type: ignore[attr-defined]
                  "their exact gain beats the last step's" if isinstance(p.bridges, gr.Straight)
                  else "")
               + (f", and patches may start apart from the street at a charge of "
                  f"{p.connectivity.island:g} of P each" if free else "")  # type: ignore[attr-defined]
               + ("; grown as a leak to ground, so its fingers run long, and carved here as an "
                  "ordinary road at ten times open ground's conductance." if grounded else
                  "; the grown ground is a road at ten times open ground's conductance."))
        return Entry(key=self.plan, name=name, how=how, cls=style[0], dot=style[1], label=name)

    def source(self) -> Source:
        return GrownSource(self.plan, budget=self.p.D)


class Ears:
    """An ears.py run (growth.py's rows)."""
    CLS = {"paths": "ear", "divides": "eard"}        # its candidates -> index.html's class

    def __init__(self, plan: str) -> None:
        import ears  # noqa: PLC0415  (its own study, still changing: imported only for its runs)
        self.plan = plan
        self.p = ears.Plan.of(re.sub(r"\.v[0-9]+$", "", plan))
        if self.p.name != plan:
            raise SystemExit(f"{plan}: ears.Plan names it {self.p.name}")

    def entry(self) -> Entry:
        cand = self.plan.split(".")[1]
        if cand not in self.CLS:
            raise SystemExit(f"{self.plan}: no page style for ears' {cand} (Ears.CLS)")
        name = f"ears, {cand}"
        how = ("Reblocks by ears: each step builds, among "
               + ("the cheapest paths from one face's edge back to it" if cand == "paths" else
                  "the divides of the escape potential across a face")
               + ", the road that lowers P most per cost (displacement, and "
               f"{price_words(self.p.cost.price)}), refusing one that does not split its face, "
               "so every road lies on a loop; grown grounded, carved here as an ordinary road at "
               "ten times.")
        return Entry(key=self.plan, name=name, how=how, cls=self.CLS[cand], dot="lkd", label=name)

    def source(self) -> Source:
        return GrownSource(self.plan, budget=self.p.D)


class JointFamily:
    """The joint SIMP's paved or roads answer beside its clearing."""

    def __init__(self, kind: str, plan: str) -> None:
        runs = Runs(rows=rs.OUT / metrics.RUNTIME, template=plan)
        self.design = Joint(kind=kind, runs=runs)
        self.kind = kind
        [price] = {rs.Plan.of(p).price for p in plan_names(runs)}
        if not isinstance(price, rs.SheetPrice):
            raise SystemExit(f"{plan}: the joint SIMP priced by the sheet (`.s<price>`)")
        self.price = price

    def entry(self) -> Entry:
        key = self.design.runs.key + f".{self.kind}"
        what = "paved" if self.kind == "paved" else "roads"
        name = f"joint SIMP, {what} + clearing, price {self.price.price:g}"
        how = ("Road SIMP with each building's clearing a second design variable, paving and "
               f"clearing priced together by the conductance they add "
               f"({price_words(self.price.price)}) under one displacement budget, "
               "solved at each budget shown: "
               + ("its paved ground (road density ½ or more) as a road at ten times"
                  if self.kind == "paved" else
                  "its paved ground's skeleton as 7 m roads linked to the street")
               + ", beside the buildings it clears (orange).")
        return Entry(key=key, name=name, how=how, cls="jp" if self.kind == "paved" else "jr",
                     dot="", label=name)

    def source(self) -> Source:
        return AnswerSource(self.design)


class PartitionFamily:
    """A partition SIMP answer (band or cut), its start named per budget where its size changes
    with D (Runs)."""
    CLS = {"cut": "ptc", "band": "ptb"}
    LABEL = {"cut": "partition SIMP, cut", "band": "partition SIMP, band, dead ends"}

    def __init__(self, answer: str, plan: str) -> None:
        if answer not in self.CLS:
            raise SystemExit(f"{answer}: a page partition answer is {' or '.join(self.CLS)}")
        import partition  # noqa: PLC0415  (its own study, still changing)
        runs = Runs(rows=partition.OUT / metrics.RUNTIME, template=plan)
        self.design = Lined(runs=runs, answer=answer)
        self.answer = answer
        plans = [partition.Plan.of(p) for p in plan_names(runs)]
        [self.price] = {p.price.price for p in plans}
        starts = [p.start for p in plans]
        if not all(isinstance(t, partition.Spectral) for t in starts):
            raise SystemExit(f"{plan}: no page words for its start (PartitionFamily: spectral)")
        at = [f"{t.n} at {D:.0%}" for t, D in zip(starts, runs.dirs, strict=True)
              if isinstance(t, partition.Spectral)]
        self.start = ("the spectral normalized cut of the buildings into clusters ("
                      + (", ".join(at[:-1]) + " and " + at[-1] if len(at) > 1 else at[0]) + ")")

    def entry(self) -> Entry:
        key = self.design.runs.key + f".{self.answer}"
        name = self.LABEL[self.answer]
        how = ("Partitions the block into labelled sub-blocks and lays the roads where the labels "
               f"change, so every road is on a loop: started from {self.start}, each sub-block a "
               "label (five, no two neighbours alike), the labels moved down the gradient of P "
               f"under the budget ({price_words(self.price)}), solved at each budget shown; "
               + ("the label boundaries as 7 m roads." if self.answer == "cut" else
                  "the band where the labels mix, as a road at ten times: the optimizer's own "
                  "roads, whose thin stretches break, leaving dead ends."))
        return Entry(key=key, name=name, how=how, cls=self.CLS[self.answer], dot="", label=name)

    def source(self) -> Source:
        return AnswerSource(self.design)


class TemplatesFamily:
    """The grid templates' best layout."""

    def __init__(self, plan: str) -> None:
        import templates  # noqa: PLC0415  (its own study, still changing)
        self.design = Lined(runs=Runs(rows=templates.OUT / metrics.RUNTIME, template=plan),
                            answer="best")

    def entry(self) -> Entry:
        key = self.design.runs.key
        name = "grid templates"
        how = ("The baseline for reblocking top-down: parallel or crossing lines at a spacing, "
               "angle and offset, each snapped to the cheapest path through the gaps and built as "
               "a 7 m road, the best layout within the budget, at each budget shown.")
        return Entry(key=key, name=name, how=how, cls="tp", dot="", label=name)

    def source(self) -> Source:
        return AnswerSource(self.design)


def family_of(argv: list[str]) -> Family:
    """The command line's family and spec (module docstring)."""
    match argv:
        case ["growth", plan]:
            return Growth(plan)
        case ["ears", plan]:
            return Ears(plan)
        case ["joint", kind, plan]:
            return JointFamily(kind, plan)
        case ["partition", answer, plan]:
            return PartitionFamily(answer, plan)
        case ["templates", plan]:
            return TemplatesFamily(plan)
    raise SystemExit(__doc__)


# ------------------------------------------------------------------ the steps

def frames(fam: Family) -> None:
    src = fam.source()
    if not src.ready():
        raise SystemExit(f"{fam.entry().key}: its rows are not finished")
    DATA.mkdir(parents=True, exist_ok=True)
    [b] = common.build_blocks([BID])
    src.frames(fam.entry().key, b)


def plates(fam: Family) -> None:
    key = fam.entry().key
    for run in fam.source().runs():
        render.plates_method(key, run.name, data=DATA, plates=render.PLATES)
        render.plates_times(key, run.name, data=DATA, plates=render.PLATES)


def page(fam: Family) -> None:
    """The method's rows and entries into index.html (replacing its own, else after the rest)."""
    e = fam.entry()
    src = fam.source()
    path = HERE / "index.html"
    html = path.read_text()
    tables = {v: local.js(html, v) for v in ("DATA", "NETS", "CLASS", "DOT", "KEY", "POINTS")}
    taken = {c: k for k, c in tables["CLASS"].items() if k != e.key}
    if e.cls in taken:
        raise SystemExit(f"{e.key}: class {e.cls} is {taken[e.cls]}'s")
    tables["DATA"][e.key] = src.rows(e.key)
    tables["NETS"][e.key] = [e.name, e.how]
    tables["CLASS"][e.key] = e.cls
    if src.points:
        tables["DOT"].pop(e.key, None)
        tables["POINTS"][e.key] = 1
    else:
        tables["DOT"][e.key] = e.dot
        tables["POINTS"].pop(e.key, None)
    item = [e.label, e.cls] + (["point"] if src.points else [])
    keys = [k for k in tables["KEY"] if k[1] != e.cls]
    tables["KEY"] = keys + [item]
    for v, value in tables.items():
        html = local.with_js(html, v, value, lines=v != "DATA")
    path.write_text(html)
    print(f"{e.key}: rows and entries into the page", flush=True)


def specs_of(d: Path) -> dict[str, Runs]:
    """The specs (with their runs) that name run folder `d`: its D as {D}, alone or with one other
    number as {n} (a start's size that changes with D), each that resolves to one run a budget."""
    spec = re.sub(r"\.D[0-9.]+\.", ".D{D}.", d.name, count=1)
    out: dict[str, Runs] = {}
    for t in [spec] + [spec[:m.start()] + "{n}" + spec[m.end():]
                       for m in re.finditer(r"[0-9]+(?:\.[0-9]+)?", spec)]:
        try:
            runs = Runs(rows=d.parent, template=t)
        except SystemExit:                  # two runs at one budget: not a spec
            continue
        if d in runs.dirs.values():
            out[t] = runs
    return out


def ready() -> None:
    """The finished runs of the families' folders not yet on the page, each with its command."""
    import ears  # noqa: PLC0415
    import partition  # noqa: PLC0415
    import templates  # noqa: PLC0415
    html = (HERE / "index.html").read_text()
    on = set(local.js(html, "DATA"))
    for run in sorted((gr.OUT / BID).iterdir()):
        is_ears = run.name.startswith("ears.")
        if (run.name.startswith("check.") or not (run / "steps.parquet").exists()
                or not run.name.endswith(f".v{ears.REV if is_ears else gr.REV}")):
            continue                        # a check, or a run of an older revision
        try:                                # an open set of folders: validated here, once
            fam: Family = Ears(run.name) if is_ears else Growth(run.name)
        except (ValueError, SystemExit) as e:
            print(f"not a current plan: {run.name} ({e})")
            continue
        rows = pd.read_parquet(run / "steps.parquet")
        hit = rows[rows.D >= 0.15]
        lp = f"L_P {float(hit.L_P.iloc[0]):.4f} at D 0.15 (eps world)" if len(hit) else ""
        if fam.source().ready() and run.name not in on:
            kind = "ears" if is_ears else "growth"
            print(f"finished: add {kind} {run.name}   {lp}")
    for kind, rows_dir in (("partition", partition.OUT), ("templates", templates.OUT),
                           ("joint", rs.OUT)):
        told: set[str] = set()
        for d in sorted((rows_dir / metrics.RUNTIME).glob("*")):
            if (d / f"{BID}.parquet").exists() and (kind != "joint" or ".cjoint." in d.name):
                specs = specs_of(d)
                if not any(k.startswith(r.key) for r in specs.values() for k in on):
                    spec = max(specs, key=lambda t: len(specs[t].dirs))    # D{D}'s on a tie
                    if spec not in told:
                        told.add(spec)
                        print(f"finished ({kind}): {d.name} -> add {kind} ... '{spec}'")


def main(argv: list[str]) -> None:
    match argv:
        case ["ready"]:
            ready()
        case ["frames", *rest]:
            frames(family_of(rest))
        case ["plates", *rest]:
            plates(family_of(rest))
        case ["page", *rest]:
            page(family_of(rest))
        case ["add", *rest]:
            fam = family_of(rest)
            frames(fam)
            plates(fam)
            page(fam)
            artifact.main()
        case _:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
