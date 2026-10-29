"""The public site's Demo page's bundle (owner 2026-10-28: "make this a new 'demo' page on the
website"; then a methods comparison on a second block). The page follows one block (PAGE, 5810):
the method's build-out on it under the slider's budgets D 0.01 .. 0.15 as escape-time plates, each
home coloured by its escape time on the block's own scale (the plain baseline's 99th percentile as
100%), short blue to long red, with isochrones every ISO_STEP of that scale (the homes' times
interpolated between their centroids, smoothed, inside the block); the screens' verdicts on it,
the shipped proxy's and the two-stage depth_density screen's; its place; the fires reported near
it; and the original topology algorithm's run time on it (a TopologyTime Strategy: extrapolated,
since it was never run there). Beside that, a comparison on a block small enough for topology to
run at all (COMPARED, 41785): both methods built out under one budget slider (a Grid each: the
shortest prefix of each method's build order reaching every budget), each frame drawn as the same
escape-time plate, with their scores and run times, topology's measured.

main cannot merge this branch, so `site <checkout>` writes it into a main checkout as a committed
bundle (SITE_DIR: demo.json, a build-out file per slider with each series' escape-time sprite
sheet, a README naming this script) and its TypeScript declaration (SITE_DTS, generated from the
dataclasses below, so a renamed field is a type error on main). Every number the page shows is in
demo.json, the fires' facts included (FIRES, quoted from their reports); the site's generator reads
it and the page's prose types none.

Reuses the build-out pipeline (atlas/buildout.py, atlas/topology_run.py, roads5810's networks.py,
compute.py, render.py): a block's frames, stored rasters and escape times are its Site's
(block_site.Site.of; no flow is recomputed here), and render.py's plate helpers. The colour map is
this script's own (SEQ).

The method is one setting (METHOD): a Shown, which names its run on each block (an OnBlock's
Pick), the reader-facing name and blurb, and how its run time is read (a Runtime Strategy).
Swapping to the partition SIMP's cut is `METHOD = PARTITION_CUT`, once networks.py `add` has
written its frames (on 5810, the networks page's block, only; the comparison then needs its Grid
on 41785 too).

    B=research/roadless/atlas/explore_proto/build.py     # from the repo root, PYTHONPATH=.
    uv run python $B plates                # the escape-time plates (minutes)
    uv run python $B site ~/src/wt_demo    # the bundle into a main checkout
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))           # research/roadless: growth, road_simp
sys.path.insert(0, str(HERE.parent / "roads5810"))
import geopandas as gpd  # noqa: E402
import growth  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networks as nw  # noqa: E402  (the build-out's sources; imports compute and render)
import pandas as pd  # noqa: E402
import render  # noqa: E402
import road_simp  # noqa: E402
import shapely  # noqa: E402
from block_site import Site  # noqa: E402  (on the path networks set)
from PIL import Image  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

cp, rnd = nw.cp, render.rnd
PLATES = HERE / "plates"                 # git-ignored: everything this script writes
FRAMES = PLATES / "frames"               # full-size plates: FRAMES/<block>/<stem>_<frame>.webp
WIDTH = 800                              # px: a sprite frame's width (roads5810's artifact.WIDTH)
QUALITY = 80                             # lossy WebP's


# ------------------------------------------------------------------ the method (one setting)

class Runtime(Protocol):
    """How long the method took to build each slider row's network, read from its own rows."""

    @property
    def machine(self) -> str:
        """Where it ran, in words (the rows do not record the host)."""
        ...

    def seconds(self, key: str, src: Any, rows: list[dict[str, Any] | None],
                site: Site) -> list[float | None]:
        """Per slider row the run time to that row's network (None where the row is none)."""
        ...


@dataclass(frozen=True)
class GrownSteps:
    """A growth or ears run (networks.GrownSource): steps.parquet's cumulative t at the row's
    stop, the state after its first m steps (the stop's prefix: networks.Grown)."""
    machine: str

    def seconds(self, key: str, src: Any, rows: list[dict[str, Any] | None],
                site: Site) -> list[float | None]:
        steps = pd.read_parquet(src.dir / "steps.parquet").set_index("step").t
        info = nw.info_of(key, "slider", data=site.networks)
        m_of = {f: m for m, f in render.frame_names(info).items()}
        return [None if r is None else float(steps[m_of[r["f"]]]) for r in rows]


@dataclass(frozen=True)
class AnswerRows:
    """An optimizer's answer per budget (networks.AnswerSource over a Lined or Joint design): the
    run's own t_total (its rows' last), each row its budget's run's."""
    machine: str

    def seconds(self, key: str, src: Any, rows: list[dict[str, Any] | None],
                site: Site) -> list[float | None]:
        design = src.design
        return [None if r is None else
                float(nw.last_row(design.dir(r["b"]) / f"{site.bid}.parquet").t_total)
                for r in rows]


class Pick(Protocol):
    """A method's stored frames on one block: its key there (the file stem in the block's
    Site.networks) and its networks.Source (runs and slider rows) on that block."""

    def key(self) -> str: ...

    def source(self, site: Site) -> Any: ...


@dataclass(frozen=True)
class GrownRun:
    """A growth or ears run by its plan (networks.GrownSource). Not through networks.Ears, which
    refuses a plan of an older ears revision: 5810's paths run is rev 1, and rev 2 (ears.REV)
    changed only `divides`."""
    plan: str
    budget: float

    def key(self) -> str:
        return self.plan

    def source(self, site: Site) -> Any:
        return nw.GrownSource(self.plan, budget=self.budget, site=site)


@dataclass(frozen=True)
class Family:
    """A networks.py family by its command line (networks.family_of). Its designs are the
    networks page's block's (networks.PAGE), so it serves that block alone."""
    argv: tuple[str, ...]

    def key(self) -> str:
        return nw.family_of(list(self.argv)).entry().key

    def source(self, site: Site) -> Any:
        if site != nw.PAGE:
            raise SystemExit(f"networks.py's families are {nw.PAGE.bid}'s, not {site.bid}'s")
        return nw.family_of(list(self.argv)).source()


@dataclass(frozen=True)
class OnBlock:
    """The method's run on one block."""
    bid: str
    pick: Pick


@dataclass(frozen=True)
class Shown:
    """The method the page shows: its runs on the page's blocks, its reader-facing name, a
    sentence on how it works, a file stem, and how its run time is read."""
    runs: tuple[OnBlock, ...]
    name: str
    blurb: str
    stem: str
    runtime: Runtime

    def pick(self, bid: str) -> Pick:
        [p] = [r.pick for r in self.runs if r.bid == bid] or [None]
        if p is None:
            raise SystemExit(f"{self.name} has no run on {bid}")
        return p


PATHS = "ears.paths.b20.gl0.25Croad.n2s0.1.M8.K4.D0.15"
EARS_PATHS = Shown(
    runs=(OnBlock("ZAF.9.3.1_1_5810", GrownRun(f"{PATHS}.v1", budget=0.15)),),
    # The owner's name for it (2026-10-29), and the coordinator's sentence for why, verbatim.
    name="Shock Reblock",
    blurb=("Each new road runs from the street, or a road already built, across the block and "
           "back to a street or road. At each step the method tries the cheapest routes through "
           "the gaps between buildings and builds the one that saves the most escape time for the "
           "homes it displaces. "
           "Named for thermal-shock cracking: as a quenched ceramic cracks inward from its cold "
           "surface and each crack ends on another, each new road here runs in from the street "
           "toward the homes with the longest way out and joins the network at both ends, so "
           "every road is on a loop."),
    stem="loop",
    runtime=GrownSteps(machine="one process on a workstation CPU (Intel Xeon w7-3455)"))

R3 = ("pt.K5.R10.band.t0.2,0.1,0.05,0.03,0.02.b1,2,4,8,16.q1,2,3,3,3.e0.2,0.1,0.05,0.02,0.i40")
PARTITION_CUT = Shown(
    runs=(OnBlock("ZAF.9.3.1_1_5810",
                  Family(("partition", "cut", f"{R3}.D{{D}}.s0.1.spec{{n}}s0.p1e0.0001v4"))),),
    name="Sub-block roads",
    blurb=("Divides the block into sub-blocks and lays a road wherever two sub-blocks meet, so "
           "every road is on a loop. The sub-blocks start from clusters of buildings and move "
           "until the roads between them save the most escape time within the budget; each "
           "budget is solved afresh."),
    stem="part",
    runtime=AnswerRows(machine="partition_rows' t_total (rows do not name the machine)"))

METHOD: Shown = EARS_PATHS


# ------------------------------------------------------------------ topology's run time

@dataclass(frozen=True)
class Timed:
    """One recorded topology run: its block, building count and seconds."""
    block: str
    n: int
    seconds: float


# topo_obj.py `time` (GeoffChurch/REU_Slum_Optimization, branch research/group-steiner,
# research/group_steiner/topo_obj.py): TopologyMethod(alpha=10, seed=0, road_width_m=7) once per
# block, single-threaded, on six of the 220 consensus blocks spaced by size; its printed output,
# recovered from the 2026-09-26 session transcript (the rows were lost with /tmp). The 11.8 h run
# is the one docs/superpowers/notes/2026-09-26-topology-is-a-group-steiner-tree.md cites.
TOPOLOGY_RUNS = (Timed("ZAF.9.3.1_1_38136", 51, 1.5), Timed("ZAF.9.3.1_1_21016", 93, 4.2),
                 Timed("ZAF.9.3.1_1_20254", 122, 14.4), Timed("ZAF.9.3.1_1_45919", 159, 14.5),
                 Timed("ZAF.9.3.1_1_44918", 257, 82.6), Timed("ZAF.9.3.1_1_5618", 759, 42477.5))
TOPOLOGY_ALPHA = 10      # the runs' TopologyMethod(alpha=...)


@dataclass(frozen=True)
class DemoTopologyExtrapolated:
    """The original topology algorithm's run time on the block, extrapolated from `runs` (each
    once, at `alpha`, on one thread) to its `n_buildings`: a least-squares power law through all
    of them (seconds ~ n^`exponent`, standard error `se`) gives `central_s`; anchored on the
    largest run, the smaller runs' exponent (`small_exponent`) gives `by_small_s` and the two
    largest runs' slope (`top_exponent`) `by_top_s`. Which of the two is the lower depends on
    which side of the largest run the block falls: above it the steeper slope gives more time,
    below it less."""
    kind: Literal["extrapolated"]
    alpha: int
    runs: tuple[Timed, ...]
    n_buildings: int
    exponent: float
    se: float
    central_s: float
    by_small_s: float
    by_top_s: float
    small_exponent: float
    top_exponent: float


@dataclass(frozen=True)
class DemoTopologyMeasured:
    """Topology (our port) run on the block itself at `alpha`, on one thread, watched as it built
    (atlas/topology_order.py): its wall time, where it ran in words, and the share of the homes'
    floor area its network displaces (it builds until every parcel touches a road); where the
    time went -- setup (`setup_s`), the first and second roads' landing times since the start,
    how many roads came after them and the median time each took; and beside it the six-run
    fit's time at this block's size (`fit_central_s`, from `fit_runs` runs), which the
    measurement checks as a further point."""
    kind: Literal["measured"]
    alpha: int
    n_buildings: int
    seconds: float
    machine: str
    displacement: float
    setup_s: float
    first_road_s: float
    second_road_s: float
    n_rest: int
    rest_median_s: float
    fit_central_s: float
    fit_runs: int


def extrapolated(runs: tuple[Timed, ...], alpha: int, n: int) -> DemoTopologyExtrapolated:
    """A least-squares power law through every run (log seconds on log n), and a range anchored on
    the largest run: the exponent of the runs below it and the slope between the two
    largest. The runs steepen with size, so the fit through all of them under-predicts the
    largest; the two anchored estimates say how far the extrapolation could go either way."""
    x = np.log([r.n for r in runs])
    y = np.log([r.seconds for r in runs])
    (b, a), *_ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)
    resid = y - (a + b * x)
    se = float(np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum()))
    (bs, _as), *_ = np.linalg.lstsq(np.vstack([x[:-1], np.ones(len(x) - 1)]).T, y[:-1],
                                    rcond=None)
    top = runs[-1]
    bt = float((y[-1] - y[-2]) / (x[-1] - x[-2]))
    return DemoTopologyExtrapolated(
        kind="extrapolated", alpha=alpha, runs=runs, n_buildings=n, exponent=float(b), se=se,
        central_s=float(np.exp(a) * n ** b), by_small_s=float(top.seconds * (n / top.n) ** bs),
        by_top_s=float(top.seconds * (n / top.n) ** bt), small_exponent=float(bs), top_exponent=bt)


class TopologyTime(Protocol):
    """Topology's run time on a block: measured there, or extrapolated to it."""

    def of(self, site: Site, n: int) -> DemoTopologyExtrapolated | DemoTopologyMeasured: ...


@dataclass(frozen=True)
class Extrapolated:
    """From the six timed runs (TOPOLOGY_RUNS) to the block's building count."""
    runs: tuple[Timed, ...]
    alpha: int

    def of(self, site: Site, n: int) -> DemoTopologyExtrapolated:
        return extrapolated(self.runs, self.alpha, n)


@dataclass(frozen=True)
class Measured:
    """The watched run on the block (atlas/topology_order.py: Site.kept/topology_built.json, and
    its segments in build order, topology_built_roads.parquet), whose frames the comparison
    shows, so its headline time and every frame's are one run's; set beside the fit through `runs`
    at the block's size. Its network is the kept run's (atlas/topology_run.py, topology.json,
    which scored it): a run that is not, or whose building count or alpha is not the page's, is
    refused."""
    runs: tuple[Timed, ...]
    alpha: int

    def of(self, site: Site, n: int) -> DemoTopologyMeasured:
        kept = json.loads((site.kept / "topology.json").read_text())
        run = json.loads((site.kept / "topology_built.json").read_text())
        if not run["same_as_kept"]:
            raise SystemExit(f"{site.bid}: topology_built.json's network is not the kept run's")
        if kept["n_buildings"] != n or run["alpha"] != self.alpha:
            raise SystemExit(f"{site.bid}: topology ran {kept['n_buildings']} buildings at alpha "
                             f"{run['alpha']}, the page {n} at {self.alpha}")
        # Each road's landing time since the start: its path's last segment's.
        roads = pd.read_parquet(site.kept / "topology_built_roads.parquet", columns=["path", "t"])
        land = roads.groupby("path")["t"].max().sort_index().to_numpy()
        if len(land) != run["n_paths"] or abs(land[0] - run["first_road_s"]) > 1e-3:
            raise SystemExit(f"{site.bid}: topology_built_roads.parquet disagrees with its record")
        each = np.diff(land)
        # The CPU as our method's machine names it, so a reader sees the same chip both times.
        cpu = str(run["cpu"]).replace("(R)", "")
        return DemoTopologyMeasured(
            kind="measured", alpha=self.alpha, n_buildings=n, seconds=float(run["wall_s"]),
            machine=f"one thread on a workstation CPU ({cpu})", displacement=float(kept["D"]),
            setup_s=round(float(run["greedy_start_s"]), 1),
            first_road_s=round(float(land[0]), 1), second_road_s=round(float(land[1]), 1),
            n_rest=len(land) - 2, rest_median_s=round(float(np.median(each[1:])), 2),
            fit_central_s=extrapolated(self.runs, self.alpha, n).central_s,
            fit_runs=len(self.runs))


@dataclass(frozen=True)
class PortSpeed:
    """atlas/topology_speed: topology's original (`original`) and our port (`port`) run on the
    same blocks, a JSON record per block and version beside the table (`dir`/table.csv)."""
    dir: Path
    original: str
    port: str

    def of(self) -> DemoPortSpeed:
        rows = list(pd.read_csv(self.dir / "table.csv", dtype={"failed": str}).itertuples())
        failed = [r for r in rows if isinstance(r.failed, str)]
        ok = [r for r in rows if not isinstance(r.failed, str)]
        if any(bool(r.censored) for r in ok):
            raise SystemExit(f"{self.dir}: a censored run; the table's speed-ups are not times")
        for r in ok:                         # the table's columns are the versions named here
            for v, col in ((self.original, r.original_s), (self.port, r.fork_s)):
                rec = json.loads((self.dir / f"{r.block}_{v}.json").read_text())
                if abs(rec["wall_s"] - col) > 1e-9:
                    raise SystemExit(f"{self.dir}: {r.block}'s {v} record disagrees with the table")
        n = np.array([r.n for r in ok], dtype=float)
        sp = np.array([r.speedup for r in ok], dtype=float)
        return DemoPortSpeed(
            port=self.port, original=self.original, n_blocks=len(ok), n_min=int(n.min()),
            n_max=int(n.max()), speedup_min=round(float(sp.min()), 2),
            speedup_median=round(float(np.median(sp)), 2), speedup_max=round(float(sp.max()), 2),
            same_networks=all(bool(r.same) for r in ok),
            exponent=round(float(np.polyfit(np.log(n), np.log(sp), 1)[0]), 2),
            n_failed=len(failed))


# The original's last commit before the performance changes, and the port's (2026-10-29).
PORT_SPEED = PortSpeed(HERE.parent / "topology_speed",
                       original="5972772", port="ff74af2")

EXTRAPOLATED = Extrapolated(TOPOLOGY_RUNS, TOPOLOGY_ALPHA)
MEASURED = Measured(TOPOLOGY_RUNS, TOPOLOGY_ALPHA)


# ------------------------------------------------------------------ the blocks

@dataclass(frozen=True)
class Block:
    """The block the page follows: its id, where it is in words, the Google Maps zoom that frames
    it, whether it is plainly an informal settlement on the ground whatever the City's survey says
    (imagery, and the news reports of fires around it), and how topology's run time on it is
    known."""
    bid: str
    place: str
    zoom: int
    plainly_informal: bool
    topology: TopologyTime


@dataclass(frozen=True)
class Grid:
    """A method's build-out on a block on one budget grid (atlas/buildout.py `table`): the
    committed per-budget table in the Site's kept files (`<key>_grid.json`: per budget a frame,
    a repeat of an earlier budget's, or stopped short of it; and the whole network), and the
    frames' corridors, damage and escape times in its networks (`<key>_grid.npz`,
    `<key>_grid_times.npz`), cut in the order the method built its roads, each frame with the
    run's time to it. `stem` names its plates; `stops_because` is what the page says when the
    method has stopped below a budget."""
    key: str
    stem: str
    stops_because: str

    def table(self, site: Site) -> dict[str, Any]:
        return json.loads((site.kept / f"{self.key}_grid.json").read_text())

    def plate(self, site: Site, frame: str) -> Path:
        return FRAMES / site.bid / f"{self.stem}_grid_{frame}.webp"

    def plates(self, site: Site, w: World) -> None:
        """The block with no new road and every frame of the grid, into FRAMES."""
        t = self.table(site)
        homes_plate(w, w.u_plain, c=np.zeros(len(w.polys)), corridor=None,
                    out=self.plate(site, "000"))
        names = {r["stop"]: r["frame"] for r in t["rows"] if r["status"] == "frame"}
        names[t["whole"]["stop"]] = t["whole"]["frame"]
        for mm, draw in stored_plates(site, w, self.key, "grid"):
            draw(self.plate(site, names[mm]))


@dataclass(frozen=True)
class Compared:
    """The block the methods are compared on: one where topology has run (a Measured time), with
    its place in words and Maps zoom, and each method's build-out on one budget grid."""
    bid: str
    place: str
    zoom: int
    topology: Measured
    ours: Grid
    earlier: Grid


PAGE = Block("ZAF.9.3.1_1_5810", "the Bloekombos and Wallacedene area of Kraaifontein", zoom=17,
             plainly_informal=True, topology=EXTRAPOLATED)
# The shipped screen's top block, and small enough (638 buildings) for topology to finish: 1.88 h
# on one thread (atlas/topology_run.py, 2026-10-29). Philippi: OpenStreetMap's suburb at its
# centre (Nominatim's reverse lookup, 2026-10-28); about 165 m across, so a closer zoom.
# Each method cut into the shortest prefixes of its own build order reaching every budget
# (atlas/buildout.py): topology's as a watched re-run built it (atlas/topology_order.py,
# 2026-10-29; the same 546 segments as the kept run), which flatters it more at low budgets than
# a standard drainage-first order did.
COMPARED = Compared(
    "ZAF.9.3.1_1_41785", "Philippi", zoom=18, topology=MEASURED,
    ours=Grid(f"{PATHS[:-len('.D0.15')]}.D0.91.v2", stem="loop",
              stops_because="no loop left to add"),
    earlier=Grid("topology.a10.s0.w7.built", stem="topology",
                 stops_because="its network is complete"))


# ------------------------------------------------------------------ colour map (OKLab)

def oklch(L: float, C: float, h: float) -> tuple[float, float, float]:
    """sRGB of an OKLCh colour (Ottosson's OKLab), refusing one out of gamut."""
    a, b = C * np.cos(np.radians(h)), C * np.sin(np.radians(h))
    l_, m_, s_ = (L + 0.3963377774 * a + 0.2158037573 * b, L - 0.1055613458 * a - 0.0638541728 * b,
                  L - 0.0894841775 * a - 1.2914855480 * b)
    lc, mc, sc = l_ ** 3, m_ ** 3, s_ ** 3
    lin = np.array([4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc,
                    -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc,
                    -0.0041960863 * lc - 0.7034186147 * mc + 1.7076147010 * sc])
    if lin.min() < -1e-3 or lin.max() > 1 + 1e-3:
        raise ValueError(f"OKLCh({L}, {C}, {h}) is out of the sRGB gamut")
    lin = np.clip(lin, 0.0, 1.0)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def ramp(name: str, stops: list[tuple[float, float, float]], n: int = 256) -> mcolors.Colormap:
    """A map through OKLCh `stops` (L, C, h), each coordinate linear between them: lightness
    linear makes equal steps of the value equal steps of perceived lightness."""
    t = np.linspace(0, 1, n)
    k = np.linspace(0, 1, len(stops))
    L, C, h = (np.interp(t, k, [s[i] for s in stops]) for i in range(3))
    return mcolors.ListedColormap([oklch(*v) for v in zip(L, C, h, strict=True)], name=name)


# one-signed (escape time): short a light blue, long a deep red, lightness falling evenly
SEQ = ramp("escape_seq", [(0.84, 0.075, 245.0), (0.62, 0.115, 320.0), (0.40, 0.150, 385.0)])
ISO_STEP = 0.10          # the isochrones' spacing, a share of the baseline's 99th percentile
ISO_GRID = 2.0           # m: the interpolated escape-time field's cell
ISO_SMOOTH = 4.0         # m: its Gaussian smoothing
DEFAULT = 5              # the slider's opening position (D 0.05)


# ------------------------------------------------------------------ plates

class World(NamedTuple):
    """What every plate of a block shares: the frame, the buildings, the block's edge and polygon,
    the plain baseline's escape times and the escape-time scale."""
    fr: Any
    polys: np.ndarray
    centroids: np.ndarray
    edge: Any
    inside: Any
    u_plain: np.ndarray
    scale: float             # the plain baseline's 99th percentile: 100% on the plate's scale


def world(site: Site) -> World:
    b = np.load(site.data / "baseline.npz", allow_pickle=True)          # our own file: WKB
    polys = shapely.from_wkb(b["polys"])
    [boundary] = shapely.from_wkb(b["boundary"])
    u = np.load(site.data / "baseline_times.npz")["u"]
    c = shapely.centroid(polys)
    return World(fr=render.frame_from(b), polys=polys,
                 centroids=np.column_stack([shapely.get_x(c), shapely.get_y(c)]),
                 edge=render.edge(b), inside=boundary, u_plain=u,
                 scale=float(np.percentile(u[np.isfinite(u)], 99)))


def field(w: World, u: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The homes' escape times (`u` where `ok`) as a field on an ISO_GRID grid over the plate:
    linear between the centroids, smoothed by ISO_SMOOTH (normalized, so the block's edge does
    not pull it toward 0), nan outside the block."""
    x0, x1, y0, y1 = w.fr.lim
    xs = np.arange(x0, x1, ISO_GRID)
    ys = np.arange(y0, y1, ISO_GRID)
    X, Y = np.meshgrid(xs, ys)
    Z = griddata(w.centroids[ok], u[ok], (X, Y), method="linear")
    have = np.isfinite(Z)
    s = ISO_SMOOTH / ISO_GRID
    num = gaussian_filter(np.where(have, Z, 0.0), s)
    den = gaussian_filter(have.astype(float), s)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.where(have & (den > 0.2), num / den, np.nan)
    Z[~shapely.contains_xy(w.inside, X, Y)] = np.nan
    return X, Y, Z


# Damage, on the escape-time plate (owner 2026-10-29: a destroyed building must not read as road).
# The damage share `c` is the build-out page's network view's (roads5810's DESTROYED ramp): 0
# untouched, 1 wholly destroyed -- which a building is once GONE_SHARE of it is taken or a road
# cuts it in two -- and partly damaged in between. A wholly destroyed building is a GHOST: a pale
# fill with a thin warm-grey outline, so the road beside it stays the only dark slate. A partly
# damaged one still has its people, so it keeps its escape-time colour and gains the same outline.
GHOST_FILL = "#ece8e2"       # a pale neutral, lighter than any escape-time stop, on the white plate
GHOST_EDGE = "#7d7468"       # the outline of every damaged building, wholly or partly
GHOST_LW = 0.7               # px at the plate's 100 dpi


def homes_plate(w: World, u: np.ndarray, *, c: np.ndarray, corridor,  # type: ignore[no-untyped-def]
                out: Path) -> None:
    """The escape-time plate (render.time_plate's figure on SEQ): each home filled by its time on
    the block's scale (nan: no home, outlined), the damage by `c` (the GHOST encoding above), the
    corridor in render.ROAD, and the isochrones every ISO_STEP of the scale, dark lines on a white
    halo."""
    f = w.fr.frame
    fig = plt.figure(figsize=(f.nx / 100, f.ny / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    norm = mcolors.Normalize(0.0, w.scale, clip=True)
    gone = c >= 1.0
    partly = (c > 0.0) & ~gone
    has = np.isfinite(u) & ~gone
    gpd.GeoSeries(list(w.polys[has])).plot(ax=ax, color=SEQ(norm(u[has])), lw=0)
    gpd.GeoSeries(list(w.polys[~has & ~gone])).boundary.plot(ax=ax, color=rnd.BUILDING, lw=0.4)
    # The ghosts under the road, so the road reads as one unbroken 7 m band and each ghost shows
    # the rest of the footprint the road took.
    if gone.any():
        gpd.GeoSeries(list(w.polys[gone])).plot(ax=ax, facecolor=GHOST_FILL,
                                                edgecolor=GHOST_EDGE, lw=GHOST_LW)
    if partly.any():
        gpd.GeoSeries(list(w.polys[partly])).boundary.plot(ax=ax, color=GHOST_EDGE, lw=GHOST_LW)
    if corridor is not None:
        gpd.GeoSeries([corridor]).plot(ax=ax, color=render.ROAD, lw=0)
    X, Y, Z = field(w, u, np.isfinite(u) & ~gone)
    levels = np.arange(ISO_STEP, 1.0 + 1e-9, ISO_STEP) * w.scale
    cs = ax.contour(X, Y, Z, levels=levels, colors=rnd.STREET, linewidths=1.1)
    cs.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white", alpha=0.85), pe.Normal()])
    render.draw_edge(ax, w.fr, w.edge)
    ax.set_xlim(w.fr.lim[0], w.fr.lim[1])
    ax.set_ylim(w.fr.lim[2], w.fr.lim[3])
    ax.set_axis_off()
    tmp = render.scratch(out)
    fig.savefig(tmp, dpi=100, facecolor="white")
    plt.close(fig)
    render.small(tmp, out)


def plate_name(site: Site, f: str) -> Path:
    return FRAMES / site.bid / f"{METHOD.stem}_{f}.webp"


def method_plates(site: Site, w: World) -> None:
    """Every frame of the method's runs on the block, into FRAMES."""
    pick = METHOD.pick(site.bid)
    key, src = pick.key(), pick.source(site)
    for run in src.runs():
        info = nw.info_of(key, run.name, data=site.networks)
        names = render.frame_names(info)
        for mm, plate in stored_plates(site, w, key, run.name):
            plate(plate_name(site, names[mm]))


def stored_plates(site: Site, w: World, key: str, run: str):  # type: ignore[no-untyped-def]
    """Per stop of a stored run (compute.frames_of's files in the Site's networks), its stop and
    a function drawing its escape-time plate to a path."""
    d = np.load(site.networks / f"{key}_{run}.npz", allow_pickle=True)          # our own: WKB
    t = np.load(site.networks / f"{key}_{run}_times.npz")
    if not np.allclose(d["frame"], np.asarray(w.fr.frame, dtype=float)):
        raise AssertionError(f"{key} {run}: its frame is not the baseline's")
    for mm in [int(s) for s in d["stops"]]:
        [corridor] = shapely.from_wkb(d[f"corridor_{mm}"])
        c, u = d[f"c_{mm}"], t[f"u_{mm}"]
        yield mm, lambda out, u=u, c=c, corridor=corridor: homes_plate(
            w, u, c=c, corridor=corridor, out=out)


def plates() -> None:
    """The page block's plain baseline and every frame of the method on it; the compared block's
    grids, both methods', all into FRAMES."""
    site = Site.of(PAGE.bid)
    plate_name(site, "000").parent.mkdir(parents=True, exist_ok=True)
    w = world(site)
    homes_plate(w, w.u_plain, c=np.zeros(len(w.polys)), corridor=None,
                out=plate_name(site, "000"))
    method_plates(site, w)
    site = Site.of(COMPARED.bid)
    w = world(site)
    for g in (COMPARED.ours, COMPARED.earlier):
        g.plates(site, w)


# ------------------------------------------------------------------ the screens

SCREEN = cp.ROOT / "examples" / "screen-map" / "capetown.json"   # the site's ScreenMap bundle
SCREEN_METRIC = "depth_density_proxy"   # the shipped screen (its floor read from the bundle)
PEEL_EXAMPLE = "depth_density"          # the two-stage screen: conf/example/depth_density.yaml


class Screen(NamedTuple):
    """The site's Cape Town screen bundle (its parsed JSON), every block's score under the shipped
    screen (gen_screen_map._score, the widget's own arithmetic) and the shipped floor's entry."""
    bundle: dict[str, Any]
    score: np.ndarray
    floor: dict[str, Any]


def screen() -> Screen:
    from scripts.gen_screen_map import _score  # noqa: PLC0415  (the repo's scripts package)

    b = json.loads(SCREEN.read_text())
    n, a, per = (np.asarray(b[k], dtype=float) for k in ("n", "area_m2", "perimeter_m"))
    [floor] = [f for f in b["floors"] if f["metric"] == SCREEN_METRIC]
    if int((_score(SCREEN_METRIC, n, a, per) >= floor["value"]).sum()) != floor["n"]:
        raise AssertionError("the screen's pool at its floor is not the bundle's")
    return Screen(bundle=b, score=_score(SCREEN_METRIC, n, a, per), floor=floor)


class Peeled(NamedTuple):
    """The two-stage screen (gen_example's, conf/example/depth_density.yaml): the shipped proxy
    keeps its top `keep_n` blocks, each is tessellated and peeled, and scored by its peel depth
    (rings of parcels in from the street to its deepest home) times its buildings per m²."""
    scores: dict[str, float]
    keep_n: int


def peeled() -> Peeled:
    """The two-stage screen's scores, from its memoized selection (DenseCompactScreen's)."""
    from hydra import compose, initialize_config_dir  # noqa: PLC0415

    from reblock.presets import load_stages  # noqa: PLC0415

    with initialize_config_dir(version_base=None, config_dir=str(cp.ROOT / "conf")):
        cfg = compose(config_name="compare_config",
                      overrides=[f"+example={PEEL_EXAMPLE}", "data=capetown_full"])
    st = load_stages(cfg)
    return Peeled(scores=st.screen.selection_scores(st.source), keep_n=st.screen.proxy_keep_n)


# ------------------------------------------------------------------ the fires

@dataclass(frozen=True)
class Report:
    """One news report: its outlet, its date (ISO), its headline (None where the page cites
    none) and its URL."""
    outlet: str
    date: str
    title: str | None
    url: str


@dataclass(frozen=True)
class Fire:
    """One fire reported near the page's block: its month (ISO, YYYY-MM), where the
    reports place it (with OpenStreetMap's distance to the block where the street can be found),
    what they say happened, and the reports. Each was checked against its article (2026-10-28);
    none is placed inside the block, and the page draws none of them on a map (owner 2026-10-28:
    some reports' streets cannot be placed)."""
    month: str
    where: str
    what: str
    reports: tuple[Report, ...]


FIRES = (
    Fire("2016-05",
         "Japhta and Dlayedwa streets, Bloekombos, Kraaifontein. OpenStreetMap puts Dlayedwa "
         "Street about 80 m outside the block.",
         "Two shacks were set alight, allegedly with a petrol bomb, and four members of one "
         "family died: a woman and three children aged 8 to 13. Neighbours tried to fight it "
         "with water and could not.",
         (Report("Cape Times", "2016-05-23", "Four die in suspected arson attack",
                 "https://iol.co.za/capetimes/news/2016-05-23-four-die-in-suspected-arson-attack/"),)),
    Fire("2020-02",
         "Wallacedene, Kraaifontein, beside the block.",
         "A fire at about 2am burned 28 shacks, displaced 83 people and killed two. Fire engines "
         "could not drive in: a community leader said “we have no roads and the shacks are "
         "closely clustered”, so crews parked on Mfeketo Street and walked to the fire. The "
         "taps gave no water, and residents pulled down shacks to stop it spreading.",
         (Report("GroundUp", "2020-02-25", "Couple dies in shack fire in Kraaifontein",
                 "https://groundup.news/article/couple-dies-shack-fire-kraaifontein/"),)),
    Fire("2022-06",
         "Pietersen Road, Wallacedene. OpenStreetMap puts Pietersen Road about 1.2 km south of "
         "the block.",
         "Three shacks burned, 11 people were displaced and three siblings died.",
         (Report("News24", "2022-06-17",
                 "Three siblings die in Cape Town fire that displaced 11 people",
                 "https://www.news24.com/southafrica/news/three-siblings-die-in-cape-town-fire-"
                 "that-displaced-11-people-20220617"),)),
    Fire("2024-10",
         "Manuel Street, Wallacedene.",
         "Four people died, three of them children, in one of 48 shack fires across Cape Town "
         "that weekend. JP Smith, the City's mayoral committee member for safety and security, "
         "said its crews arrive quickly but are “hampered by other factors such as "
         "access”; he spoke of the weekend's fires as a whole.",
         (Report("TimesLIVE", "2024-10-28",
                 "Nine people perish in weekend informal settlement fires in Cape Town",
                 "https://www.timeslive.co.za/news/south-africa/2024-10-28-nine-people-perish-in-"
                 "weekend-informal-settlement-fires-in-cape-town/"),)),
    Fire("2025-07",
         "Maroela Road, Wallacedene. OpenStreetMap puts the Wallacedene stretch of Maroela Road "
         "about 630 m south of the block; the road also passes within about 16 m of the block's "
         "south edge.",
         "About 20 shacks burned and 60 to 80 people were displaced. A 6-year-old girl died.",
         (Report("EWN", "2025-07-15", None,
                 "https://www.ewn.co.za/2025/07/15/kraaifontein-police-monitoring-wallacedene-"
                 "after-shack-fire-claims-life-of-girl"),
          Report("Cape Argus", "2025-07-17", None,
                 "https://capeargus.co.za/news/2025-07-17-tragic-shack-fire-claims-life-of-6-"
                 "year-old-girl-in-wallacedene/"))),
)


# ------------------------------------------------------------------ the bundle

SITE_DIR = Path("examples") / "demo"            # in a main checkout: the bundle
SITE_DTS = Path("web") / "src" / "demo.d.ts"   # in a main checkout: the bundle's declaration
# A building counts as wholly destroyed once this share of it is gone (or a road cuts it in two):
# the displacement rule the build-out page's network plate states (roads5810's caption).
GONE_SHARE = 0.1


@dataclass(frozen=True)
class DemoProxy:
    """The block under the shipped screen: its buildings `n`, area and perimeter (the three
    numbers the score is made of), its score, its rank (1 the highest) among all of the city's
    blocks, and whether the City's informal-settlement survey flags it."""
    n: int
    area_m2: float
    perimeter_m: float
    score: float
    rank: int
    informal: bool


@dataclass(frozen=True)
class DemoPeel:
    """The block under the two-stage depth_density screen: its score (`rings`, its peel depth --
    rings of parcels from the street to its deepest home -- times buildings per m²) and its rank
    among the blocks that screen scores (DemoScreen.peel_keep_n, the proxy's top)."""
    score: float
    rank: int
    rings: int


@dataclass(frozen=True)
class DemoStep:
    """A network the build-out shows: the share of the homes' floor area it displaces, its new
    road, its L_P (`permeability`), the conductance it adds (m² of open ground), the method's run
    time to it since the run began (`seconds`), and its frame's row in the series' sheet."""
    displacement: float
    road_km: float
    permeability: float
    conductance_m2: float
    seconds: float
    row: int


@dataclass(frozen=True)
class DemoMet:
    """The budget met: the shortest stretch of the method's build order that displaces at least
    the budget (`step`; one road can overshoot several budgets, so several can share a step)."""
    kind: Literal["met"]
    step: DemoStep


@dataclass(frozen=True)
class DemoShort:
    """The method stopped below the budget: `step` is its whole network, displacing less, and
    `why` says why it stopped, in words."""
    kind: Literal["short"]
    step: DemoStep
    why: str


@dataclass(frozen=True)
class DemoRecorded:
    """A method's whole run time, recorded by its own run, and where it ran in words."""
    kind: Literal["recorded"]
    seconds: float
    machine: str


@dataclass(frozen=True)
class DemoSheet:
    """A series' escape-time plates as one sprite sheet beside its build-out file (`file`):
    `frames` frames of `width` x `frame_height` px in `columns` columns, read across then down
    (a step's `row` is its frame's index), the block with no new road first. Columns keep a long
    series under WebP's 16383 px."""
    file: str
    width: int
    frame_height: int
    frames: int
    columns: int


@dataclass(frozen=True)
class DemoSeries:
    """One method on the build-out: its name, the label the page heads it with, its plates, its
    whole run time (recorded, or measured for topology), and what it shows at each of the
    build-out's budgets, each cut in the order the method built its roads."""
    method: str
    label: str
    sheet: DemoSheet
    run: DemoRecorded | DemoTopologyMeasured
    cells: tuple[DemoMet | DemoShort, ...]


@dataclass(frozen=True)
class DemoBuildOut:
    """One block's build-out under one budget slider: its escape-time scale (the model's units at
    its plain baseline's 99th-percentile home, 100% on every plate), the slider's budgets (0 the
    block with no new road), the position it opens on, and one series per method shown, each
    with a cell per budget; and the plates' encoding (DemoBundle's, repeated so the widget needs
    no second file). Written beside demo.json as its own file, which the page's build-out widget
    fetches."""
    block_id: str
    escape_scale: float
    budgets: tuple[float, ...]
    default_step: int
    series: tuple[DemoSeries, ...]
    encoding: DemoEncoding


@dataclass(frozen=True)
class DemoBlock:
    """The block the page follows: its id, buildings, place in words, centroid and a Google Maps
    link; whether it is plainly an informal settlement on the ground (Block's), whatever the
    survey (`proxy.informal`) says; the screens' verdicts on it; the fires reported near it; its
    build-out's file (`buildout`); and the original topology algorithm's run time on it,
    measured or extrapolated (`kind`)."""
    block_id: str
    n_buildings: int
    place: str
    lat: float
    lon: float
    maps_url: str
    plainly_informal: bool
    proxy: DemoProxy
    peel: DemoPeel
    fires: tuple[Fire, ...]
    buildout: str
    topology: DemoTopologyExtrapolated | DemoTopologyMeasured


@dataclass(frozen=True)
class DemoComparison:
    """The methods compared on a block where topology has run: the block (id, buildings, place,
    Maps link, its rank under the shipped screen), its build-out's file (`buildout`: our method
    and topology on one slider), and topology's measured run there."""
    block_id: str
    n_buildings: int
    place: str
    maps_url: str
    proxy_rank: int
    buildout: str
    topology: DemoTopologyMeasured


@dataclass(frozen=True)
class DemoScreen:
    """The screens, city-wide: the shipped proxy (`metric`) at its `floor`, its `pool`, and that
    pool's precision and recall against the City's survey, over `n_blocks`; and the two-stage
    screen (`peel_metric`), which peels the proxy's top `peel_keep_n`."""
    metric: str
    floor: float
    pool: int
    precision: float
    recall: float
    n_blocks: int
    peel_metric: str
    peel_keep_n: int


@dataclass(frozen=True)
class DemoMethod:
    """The method the page builds out: its reader-facing name and how it works."""
    name: str
    blurb: str


@dataclass(frozen=True)
class DemoEncoding:
    """The plates' colours and the quantities their legend states. `escape_ramp` maps escape time
    from 0 to 100% of a block's `escape_scale`, short blue to long red, with an isochrone every
    `iso_step` of it. `road` is a new road's corridor, `road_width_m` wide, conducting
    `road_conductance` times what open ground does. A wholly destroyed building -- once
    `gone_share` of it is gone, or a road cuts it in two -- is a ghost (`ghost_fill`, outlined
    `ghost_edge`); a partly damaged one keeps its escape-time colour and the same outline. The
    block's edge: `street` (where people walk out), `edge` (a railway's or a ditch's line),
    `water` (a ditch).
    `depth_ramp` is the peel figure's (reblock.render's depth colouring, ring 1 to the deepest),
    evenly spaced stops."""
    road: str
    street: str
    edge: str
    water: str
    escape_ramp: tuple[str, ...]
    iso_step: float
    road_width_m: float
    road_conductance: float
    gone_share: float
    ghost_fill: str
    ghost_edge: str
    depth_ramp: tuple[str, ...]


@dataclass(frozen=True)
class DemoPortSpeed:
    """Topology's run times on the page are our optimized port's (`port`, a commit of
    GeoffChurch's topology fork); the original (`original`, the last commit before the
    performance changes) timed against it on `n_blocks` blocks of `n_min` to `n_max` buildings:
    how many times as long the original took (least, median, most), whether every network was
    the same, the slope of log speed-up on log n (`exponent`), and how many blocks failed in both
    and are left out (atlas/topology_speed/table.csv)."""
    port: str
    original: str
    n_blocks: int
    n_min: int
    n_max: int
    speedup_min: float
    speedup_median: float
    speedup_max: float
    same_networks: bool
    exponent: float
    n_failed: int


@dataclass(frozen=True)
class DemoBundle:
    """The Demo page's numbers: the block it follows, the methods compared on another, the
    screens, the method, the plates' encoding, and how topology's times relate to the original
    algorithm's (`topology_port`)."""
    block: DemoBlock
    comparison: DemoComparison
    screen: DemoScreen
    method: DemoMethod
    encoding: DemoEncoding
    topology_port: DemoPortSpeed


def ts_type(t: Any) -> str:
    """`t`'s TypeScript spelling, for the bundle's types only (refusing anything else)."""
    if t is str:
        return "string"
    if t is int or t is float:
        return "number"
    if t is bool:
        return "boolean"
    if typing.get_origin(t) is Literal:
        return " | ".join(json.dumps(a) for a in typing.get_args(t))
    if typing.get_origin(t) is tuple:
        args = typing.get_args(t)
        if len(args) == 2 and args[1] is Ellipsis:
            item = ts_type(args[0])
            return f"({item})[]" if " | " in item else f"{item}[]"
        return "[" + ", ".join(ts_type(a) for a in args) + "]"
    if typing.get_origin(t) is types.UnionType:
        return " | ".join("null" if a is type(None) else ts_type(a) for a in typing.get_args(t))
    if dataclasses.is_dataclass(t):
        return t.__name__
    raise TypeError(f"no TypeScript spelling for {t!r}")


def dts(*tops: type) -> str:
    """`tops` and every dataclass they hold, however deep in a tuple or a union, as TypeScript
    interfaces, each documented by its docstring, the held ones first."""
    order: list[type] = []

    def held(t: Any) -> list[type]:
        if dataclasses.is_dataclass(t):
            return [t]  # type: ignore[list-item]
        return [d for a in typing.get_args(t) for d in held(a)]

    def visit(cls: type) -> None:
        for hint in typing.get_type_hints(cls).values():
            for t in held(hint):
                if t not in order:
                    visit(t)
        if cls not in order:
            order.append(cls)

    for top in tops:
        visit(top)
    out = ["// GENERATED by research/roadless/atlas/explore_proto/build.py (branch",
           "// research/roadless) -- do not edit. Regenerate there: build.py site <this checkout>",
           "// This file is what makes a renamed Python field a TypeScript error instead of a",
           "// blank panel."]
    for cls in order:
        doc = (cls.__doc__ or "").strip().splitlines()
        out.append("/** " + "\n * ".join(line.strip() for line in doc) + " */")
        out.append(f"export interface {cls.__name__} {{")
        out += [f"  {k}: {ts_type(t)};" for k, t in typing.get_type_hints(cls).items()]
        out.append("}")
    return "\n".join(out) + "\n"


SITE_README = """\
<!-- GENERATED by research/roadless/atlas/explore_proto/build.py (branch research/roadless) --
     do not edit. Regenerate there: build.py site <a main checkout> -->

# The Demo page's bundle

What the site's Demo page reads: Cape Town block `{bid}`, followed from the city-wide screens to
the roads one method builds through it, and that method beside the original topology algorithm
on block `{cid}`, where topology has run.

- `demo.json`: the page's facts -- the block's place, the shipped screen's verdict on it (checked
  against `../screen-map/capetown.json`) and the two-stage depth_density screen's, the fires
  reported near it (each quoted from its report and checked against the article; none is placed
  inside the block, and the page draws none of them on a map), topology's run time on each block
  (extrapolated on the first, measured on the second), and the encoding.
- `{page}` and `{compared}`: the two build-outs the page's slider widgets fetch -- budgets, and per
  method its escape-time sprite sheet (beside them) and what it shows at each budget.

`web/src/demo.d.ts` is generated beside it from the same dataclasses, so a renamed field is a
TypeScript error.

The producer is `research/roadless/atlas/explore_proto/build.py` on the `research/roadless`
branch, which cannot merge into `main` (it carries a private submit group and a large GPU group):
the frames are the build-out pipeline's stored ones, so this directory is how they reach the
public site.
"""


def sheet_of(paths: list[Path], out: Path, width: int) -> DemoSheet:
    """The plates `paths` at `width` px as one sprite sheet into `out`, in as few columns as keep
    it under WebP's 16383 px."""
    ims = [Image.open(p) for p in paths]
    ims = [im.resize((width, round(im.height * width / im.width)), Image.LANCZOS) for im in ims]
    h = ims[0].height
    if any(i.height != h for i in ims):
        raise AssertionError(f"{out.name}: frames of different heights")
    per = 16383 // h                                         # frames down one column
    cols = -(-len(ims) // per)
    down = -(-len(ims) // cols)
    sheet = Image.new("RGB", (width * cols, h * down), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * width, (i // cols) * h))
    sheet.save(out, quality=QUALITY, method=6)
    return DemoSheet(file=out.name, width=width, frame_height=h, frames=len(ims), columns=cols)


def maps_url(lat: float, lon: float, zoom: int) -> str:
    """Google Maps' satellite view centred on (lat, lon)."""
    return (f"https://www.google.com/maps/@?api=1&map_action=map&center={lat:.5f}%2C{lon:.5f}"
            f"&zoom={zoom}&basemap=satellite")


def centroid_wgs84(bid: str) -> tuple[float, float]:
    """The block's centroid (its own polygon, in its CRS) as (lat, lon)."""
    [b] = cp.common.build_blocks([bid])
    c = shapely.centroid(b.boundary)
    lon, lat = Transformer.from_crs(b.crs, "EPSG:4326", always_xy=True).transform(c.x, c.y)
    return float(lat), float(lon)


def cells_of(site: Site, budgets: tuple[float, ...]
             ) -> tuple[list[DemoMet | DemoShort], list[str]]:
    """The method's cells on the block at `budgets` (0 the plain baseline, then the slider's: the
    stored run's rows, render.SLIDER's), and its sheet's frame names in row order. Budgets the
    stored run has no row for are refused: the page states what the method did, never pads it."""
    pick = METHOD.pick(site.bid)
    key, src = pick.key(), pick.source(site)
    rows = src.rows(key)
    secs = METHOD.runtime.seconds(key, src, rows, site)
    by_budget = dict(zip(nw.SLIDER, zip(rows, secs, strict=True), strict=True))
    fs = ["000"] + list(dict.fromkeys(r["f"] for r in rows if r is not None))
    cells: list[DemoMet | DemoShort] = []
    for t in budgets:
        if t == 0.0:
            cells.append(DemoMet(kind="met", step=BASELINE_STEP))
            continue
        if t not in by_budget:
            raise SystemExit(f"{site.bid}: the stored run has no row at the {t:g} budget")
        r, sec = by_budget[t]
        if r is None:
            raise SystemExit(f"{site.bid}: the stored run has no network at the {t:g} budget")
        if r.get("full", False):
            raise SystemExit(f"{site.bid}: the stored run stopped below the {t:g} budget, which "
                             f"the page's slider offers")
        cells.append(DemoMet(kind="met", step=DemoStep(
            displacement=r["D"], road_km=r["km"], permeability=r["LP"],
            conductance_m2=float(r["C"]), seconds=sec, row=fs.index(r["f"]))))
    return cells, fs


# The block with no new road: every series' first row.
BASELINE_STEP = DemoStep(displacement=0.0, road_km=0.0, permeability=0.0, conductance_m2=0.0,
                         seconds=0.0, row=0)
PAGE_BUDGETS = (0.0, *nw.SLIDER)


def method_series(site: Site, budgets: tuple[float, ...], label: str,
                  out: Path) -> DemoSeries:
    """The method on the block at `budgets`, its sheet into `out`."""
    cells, fs = cells_of(site, budgets)
    whole = max(c.step.seconds for c in cells)
    return DemoSeries(method=METHOD.name, label=label,
                      sheet=sheet_of([plate_name(site, f) for f in fs],
                                     out / f"{METHOD.stem}_{site.bid}.webp", WIDTH),
                      run=DemoRecorded(kind="recorded", seconds=whole,
                                       machine=METHOD.runtime.machine),
                      cells=tuple(cells))


def write_buildout(out: Path, b: DemoBuildOut) -> str:
    name = f"buildout_{b.block_id}.json"
    (out / name).write_text(json.dumps(dataclasses.asdict(b), indent=1, ensure_ascii=False)
                            + "\n")
    return name


def proxy_of(sc: Screen, bid: str) -> DemoProxy:
    b = sc.bundle
    i = b["block_id"].index(bid)
    return DemoProxy(n=int(b["n"][i]), area_m2=float(b["area_m2"][i]),
                     perimeter_m=float(b["perimeter_m"][i]), score=float(sc.score[i]),
                     rank=int((sc.score > sc.score[i]).sum()) + 1,
                     informal=bool(b["informal"][i]))


def page_block(sc: Screen, pl: Peeled, out: Path, enc: DemoEncoding) -> DemoBlock:
    """Everything the page shows about PAGE; its build-out (one series, the method) into `out`."""
    site = Site.of(PAGE.bid)
    proxy = proxy_of(sc, PAGE.bid)
    peel_score = pl.scores[PAGE.bid]
    rings = peel_score * proxy.area_m2 / proxy.n          # depth_density = peel depth x n/A
    # The bundle rounds areas (to 0.1 m²), so the quotient lands within a hair of a whole number;
    # another block's n or A would miss it by far more.
    if abs(rings - round(rings)) > 1e-3:
        raise AssertionError(f"{PAGE.bid}: depth_density {peel_score} over n/A is not a whole "
                             f"number of rings ({rings}); the bundle's n or A is not the screen's")
    buildout = write_buildout(out, DemoBuildOut(
        block_id=PAGE.bid, escape_scale=round(world(site).scale, 1), budgets=PAGE_BUDGETS,
        default_step=DEFAULT,
        series=(method_series(site, PAGE_BUDGETS, f"Our method: {METHOD.name}", out),),
        encoding=enc))
    lat, lon = centroid_wgs84(PAGE.bid)
    return DemoBlock(
        block_id=PAGE.bid, n_buildings=proxy.n, place=PAGE.place, lat=lat, lon=lon,
        maps_url=maps_url(lat, lon, PAGE.zoom), plainly_informal=PAGE.plainly_informal,
        proxy=proxy,
        peel=DemoPeel(score=float(peel_score),
                      rank=1 + sum(1 for v in pl.scores.values() if v > peel_score),
                      rings=round(rings)),
        fires=FIRES, buildout=buildout, topology=PAGE.topology.of(site, proxy.n))


def grid_series(site: Site, w: World, g: Grid, budgets: tuple[float, ...], *, method: str,
                label: str, run: DemoRecorded | DemoTopologyMeasured, out: Path) -> DemoSeries:
    """The method's grid on the compared block at `budgets` (0 the plain baseline, then the
    table's grid, then any whole networks' displacements past it), its sheet into `out`."""
    t = g.table(site)
    if abs(t["escape_scale"] - w.scale) > 1e-6 * w.scale:
        raise SystemExit(f"{g.key}: its escape times are on scale {t['escape_scale']}, the "
                         f"block's baseline {w.scale}")
    by_budget = {r["budget"]: r for r in t["rows"]}
    fs = ["000"] + [r["frame"] for r in t["rows"] if r["status"] == "frame"]
    if t["whole"]["frame"] not in fs:
        fs.append(t["whole"]["frame"])

    def step(r: dict[str, Any]) -> DemoStep:
        return DemoStep(displacement=r["D"], road_km=r["km"], permeability=r["L_P"],
                        conductance_m2=float(r["C"]), seconds=float(r["seconds"]),
                        row=fs.index(r["frame"]))

    whole = step(t["whole"])
    cells: list[DemoMet | DemoShort] = []
    for b in budgets:
        if b == 0.0:
            cells.append(DemoMet(kind="met", step=BASELINE_STEP))
        elif b not in by_budget:                    # past the grid: one whole network's budget
            cells.append(DemoMet(kind="met", step=whole) if whole.displacement >= b
                         else DemoShort(kind="short", step=whole, why=g.stops_because))
        else:
            r = by_budget[b]
            match r["status"]:
                case "frame":
                    cells.append(DemoMet(kind="met", step=step(r)))
                case "repeat":
                    cells.append(DemoMet(kind="met", step=step(by_budget[r["repeat_of"]])))
                case "stopped":
                    cells.append(DemoShort(kind="short", step=whole, why=g.stops_because))
                case other:
                    raise SystemExit(f"{g.key}: unknown status {other!r} at {b}")
    return DemoSeries(method=method, label=label,
                      sheet=sheet_of([g.plate(site, f) for f in fs],
                                     out / f"{g.stem}_{site.bid}.webp", COMPARE_WIDTH),
                      run=run, cells=tuple(cells))


# px: a compared plate's width -- two sit side by side, each at most about half the page's
COMPARE_WIDTH = 600


def comparison(sc: Screen, out: Path, enc: DemoEncoding) -> DemoComparison:
    """The methods compared on COMPARED under one slider, their sheets and the build-out file
    into `out`. The budgets are the grid's (both methods' tables share it), then topology's whole
    network's displacement, where topology ends; the slider opens on the last budget both methods
    meet."""
    site = Site.of(COMPARED.bid)
    w = world(site)
    proxy = proxy_of(sc, COMPARED.bid)
    whole = COMPARED.topology.of(site, proxy.n)
    ours_t, earlier_t = COMPARED.ours.table(site), COMPARED.earlier.table(site)
    if ours_t["grid"] != earlier_t["grid"]:
        raise SystemExit(f"{COMPARED.bid}: the two methods' grids differ")
    budgets = (0.0, *ours_t["grid"], earlier_t["whole"]["D"])
    ours = grid_series(site, w, COMPARED.ours, budgets, method=METHOD.name,
                       label=f"Our method: {METHOD.name}",
                       run=DemoRecorded(kind="recorded", seconds=ours_t["whole"]["seconds"],
                                        machine=METHOD.runtime.machine), out=out)
    earlier = grid_series(site, w, COMPARED.earlier, budgets, method="topology",
                          label="Earlier method: topology (our optimized port)", run=whole, out=out)
    both = [k for k, (a, b) in enumerate(zip(ours.cells, earlier.cells, strict=True))
            if isinstance(a, DemoMet) and isinstance(b, DemoMet) and k > 0]
    buildout = write_buildout(out, DemoBuildOut(
        block_id=COMPARED.bid, escape_scale=round(w.scale, 1), budgets=budgets,
        default_step=both[-1], series=(ours, earlier), encoding=enc))
    lat, lon = centroid_wgs84(COMPARED.bid)
    return DemoComparison(
        block_id=COMPARED.bid, n_buildings=proxy.n, place=COMPARED.place,
        maps_url=maps_url(lat, lon, COMPARED.zoom), proxy_rank=proxy.rank, buildout=buildout,
        topology=whole)


def site(root: Path) -> None:
    """The Demo page's bundle (SITE_DIR) and its declaration (SITE_DTS) into the main checkout
    `root`."""
    from reblock import render as reblock_render  # noqa: PLC0415  (the depth figure's colouring)

    if not (root / "mkdocs.yml").is_file() or not (root / "web" / "src").is_dir():
        raise SystemExit(f"{root} is not a checkout of the site (no mkdocs.yml or web/src)")
    out = root / SITE_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sc, pl = screen(), peeled()
    depth = matplotlib.colormaps[reblock_render._CMAP]
    enc = DemoEncoding(
        road=render.ROAD, street=rnd.STREET, edge=rnd.EDGE, water=nw.cp.gm.WATER_FILL,
        escape_ramp=tuple(mcolors.to_hex(SEQ(t)) for t in np.linspace(0, 1, 11)),
        iso_step=ISO_STEP, road_width_m=road_simp.ROAD_W, road_conductance=growth.ROAD_KAPPA,
        gone_share=GONE_SHARE, ghost_fill=GHOST_FILL, ghost_edge=GHOST_EDGE,
        depth_ramp=tuple(mcolors.to_hex(depth(t)) for t in np.linspace(0, 1, 9)))
    block, compared = page_block(sc, pl, out, enc), comparison(sc, out, enc)
    floor = sc.floor
    bundle = DemoBundle(
        block=block, comparison=compared,
        screen=DemoScreen(metric=SCREEN_METRIC, floor=float(floor["value"]), pool=int(floor["n"]),
                          precision=float(floor["precision"]), recall=float(floor["recall"]),
                          n_blocks=len(sc.score), peel_metric=PEEL_EXAMPLE,
                          peel_keep_n=pl.keep_n),
        method=DemoMethod(name=METHOD.name, blurb=METHOD.blurb),
        topology_port=PORT_SPEED.of(),
        encoding=enc)
    (out / "demo.json").write_text(json.dumps(dataclasses.asdict(bundle), indent=1,
                                              ensure_ascii=False) + "\n")
    (out / "README.md").write_text(SITE_README.format(
        bid=block.block_id, cid=compared.block_id, page=block.buildout, compared=compared.buildout))
    (root / SITE_DTS).write_text(dts(DemoBundle, DemoBuildOut))
    mb = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"{out}: {len(list(out.iterdir()))} files, {mb:.1f} MB; {root / SITE_DTS}", flush=True)
    print(f"  {block.block_id}: proxy #{block.proxy.rank} {block.proxy.score:.4f}; peel "
          f"#{block.peel.rank} {block.peel.score:.4f} ({block.peel.rings} rings); topology "
          f"{block.topology.kind}; compared on {compared.block_id}, topology D "
          f"{compared.topology.displacement:.3f} in {compared.topology.seconds:.0f} s",
          flush=True)


def main(argv: list[str]) -> None:
    matplotlib.use("Agg")
    match argv:
        case ["plates"]:
            plates()
        case ["site", root]:
            site(Path(root).expanduser().resolve())
        case _:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
