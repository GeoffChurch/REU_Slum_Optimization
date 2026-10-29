"""The public site's Demo page's bundle (owner 2026-10-28: "make this a new 'demo' page on the
website"; then a methods comparison on a second block). The page follows one block (PAGE, 5810):
the method's build-out on it under the slider's budgets D 0.01 .. 0.15 as escape-time plates, each
home coloured by its escape time on the block's own scale (the plain baseline's 99th percentile as
100%), short blue to long red, with isochrones every ISO_STEP of that scale (the homes' times
interpolated between their centroids, smoothed, inside the block); the screens' verdicts on it,
the shipped proxy's and the two-stage depth_density screen's; its place; the fires reported near
it; and the original topology algorithm's run time on it (a TopologyTime Strategy: extrapolated,
since it was never run there). Beside that, a comparison on a block small enough for topology to
run at all (COMPARED, 41785): the method's full build-out and topology's network, each drawn as
the same escape-time plate, with their scores and run times, topology's measured.

main cannot merge this branch, so `site <checkout>` writes it into a main checkout as a committed
bundle (SITE_DIR: demo.json, the page block's escape-time sprite sheet, the comparison's two
plates, a README naming this script) and its TypeScript declaration (SITE_DTS, generated from the
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
written its frames (on 5810, the networks page's block, only; the comparison then needs one on
41785 too).

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
    runs=(OnBlock("ZAF.9.3.1_1_5810", GrownRun(f"{PATHS}.v1", budget=0.15)),
          OnBlock("ZAF.9.3.1_1_41785", GrownRun(f"{PATHS}.v2", budget=0.15))),
    name="Loop roads",
    blurb=("Each new road runs from the street, or a road already built, across the block and "
           "back to a street or road, so every road is on a loop and none is a dead end. At each "
           "step the method tries the cheapest routes through the gaps between buildings and "
           "builds the one that saves the most escape time for the homes it displaces."),
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
    """The original topology algorithm run on the block itself, once, at `alpha`, on one thread
    (atlas/topology_run.py): its wall time, where it ran in words, and the share of the homes'
    floor area its network displaces (it builds until every parcel touches a road); beside it the
    six-run fit's time at this block's size (`fit_central_s`, from `fit_runs` runs), which the
    measurement checks as a further point."""
    kind: Literal["measured"]
    alpha: int
    n_buildings: int
    seconds: float
    machine: str
    displacement: float
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
    """atlas/topology_run.py's run on the block (Site.kept/topology.json, its roads scored there
    too), set beside the fit through `runs` at the block's size; refuses a run whose building
    count or alpha is not the page's, or whose roads were never scored."""
    runs: tuple[Timed, ...]
    alpha: int

    def of(self, site: Site, n: int) -> DemoTopologyMeasured:
        info = json.loads((site.kept / "topology.json").read_text())
        if info["n_buildings"] != n or info["alpha"] != self.alpha:
            raise SystemExit(f"{site.bid}: topology.json ran {info['n_buildings']} buildings at "
                             f"alpha {info['alpha']}, the page {n} at {self.alpha}")
        if "D" not in info:
            raise SystemExit(f"{site.bid}: topology.json's roads are not scored (topology_run.py "
                             f"score)")
        # The CPU as Loop roads' machine names it, so a reader sees the same chip both times.
        cpu = str(info["cpu"]).replace("(R)", "")
        return DemoTopologyMeasured(
            kind="measured", alpha=self.alpha, n_buildings=n, seconds=float(info["wall_s"]),
            machine=f"one thread on a workstation CPU ({cpu})", displacement=float(info["D"]),
            fit_central_s=extrapolated(self.runs, self.alpha, n).central_s,
            fit_runs=len(self.runs))

    def network(self, site: Site) -> tuple[str, str]:
        """The run's scored network in the Site's networks: (key, run), as topology.json names
        them (topology_run.py `score`)."""
        info = json.loads((site.kept / "topology.json").read_text())
        return info["scored_key"], info["scored_run"]

    def scores(self, site: Site) -> tuple[float, float, float]:
        """The scored network's displacement, L_P and new road (km), topology.json's."""
        info = json.loads((site.kept / "topology.json").read_text())
        return float(info["D"]), float(info["L_P"]), float(info["road_m"]) / 1000


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
class Compared:
    """The block the methods are compared on: one where topology has run (a Measured time, whose
    record names its scored network), with its place in words and Maps zoom."""
    bid: str
    place: str
    zoom: int
    topology: Measured


PAGE = Block("ZAF.9.3.1_1_5810", "the Bloekombos and Wallacedene area of Kraaifontein", zoom=17,
             plainly_informal=True, topology=EXTRAPOLATED)
# The shipped screen's top block, and small enough (638 buildings) for topology to finish: 1.88 h
# on one thread (atlas/topology_run.py, 2026-10-29). Philippi: OpenStreetMap's suburb at its
# centre (Nominatim's reverse lookup, 2026-10-28); about 165 m across, so a closer zoom.
COMPARED = Compared("ZAF.9.3.1_1_41785", "Philippi", zoom=18, topology=MEASURED)


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


def homes_plate(w: World, u: np.ndarray, *, gone: np.ndarray, corridor,  # type: ignore[no-untyped-def]
                out: Path) -> None:
    """The escape-time plate (render.time_plate's figure on SEQ): each home filled by its time on
    the block's scale (nan: no home, outlined), the gone buildings and the corridor in
    render.ROAD, and the isochrones every ISO_STEP of the scale, dark lines on a white halo."""
    f = w.fr.frame
    fig = plt.figure(figsize=(f.nx / 100, f.ny / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    norm = mcolors.Normalize(0.0, w.scale, clip=True)
    has = np.isfinite(u) & ~gone
    gpd.GeoSeries(list(w.polys[has])).plot(ax=ax, color=SEQ(norm(u[has])), lw=0)
    gpd.GeoSeries(list(w.polys[~has & ~gone])).boundary.plot(ax=ax, color=rnd.BUILDING, lw=0.4)
    gpd.GeoSeries(list(w.polys[gone])).plot(ax=ax, color=render.ROAD, lw=0)
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


def topology_plate(site: Site) -> Path:
    return FRAMES / site.bid / "topology.webp"


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
        gone, u = d[f"c_{mm}"] >= 1.0, t[f"u_{mm}"]
        yield mm, lambda out, u=u, gone=gone, corridor=corridor: homes_plate(
            w, u, gone=gone, corridor=corridor, out=out)


def plates() -> None:
    """The page block's plain baseline and every frame of the method on it; the compared block's
    method frames and topology's scored network, all into FRAMES."""
    site = Site.of(PAGE.bid)
    plate_name(site, "000").parent.mkdir(parents=True, exist_ok=True)
    w = world(site)
    homes_plate(w, w.u_plain, gone=np.zeros(len(w.polys), dtype=bool), corridor=None,
                out=plate_name(site, "000"))
    method_plates(site, w)
    site = Site.of(COMPARED.bid)
    plate_name(site, "000").parent.mkdir(parents=True, exist_ok=True)
    w = world(site)
    method_plates(site, w)
    key, run = COMPARED.topology.network(site)
    [(_, plate)] = list(stored_plates(site, w, key, run))
    plate(topology_plate(site))


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
class DemoFrame:
    """One slider position: the network the method has built by the displacement budget `budget`
    (a share of the homes' floor area; 0 the block with no new road). `displacement` is the share
    it reaches (at least `budget`; a run's step can overshoot several budgets at once, so several
    positions can share one network), `road_km` its new road, `permeability` its L_P,
    `conductance_m2` the conductance it adds (m² of open ground), `seconds` the method's run time
    to it, `row` its frame's row in the block's sprite sheet."""
    budget: float
    displacement: float
    road_km: float
    permeability: float
    conductance_m2: float
    seconds: float
    row: int


@dataclass(frozen=True)
class DemoSheet:
    """The block's escape-time plates as one sprite sheet beside demo.json (`file`): `rows` frames
    of `width` x `frame_height` px stacked top to bottom."""
    file: str
    width: int
    frame_height: int
    rows: int


@dataclass(frozen=True)
class DemoBlock:
    """The block the page follows: its id, buildings, place in words, centroid and a Google Maps
    link; whether it is plainly an informal settlement on the ground (Block's), whatever the
    survey (`proxy.informal`) says; the screens' verdicts on it; the fires reported near it; the
    method's build-out on it (its frames, its plates, the escape-time scale -- the model's units
    at the block's plain baseline's 99th-percentile home, 100% on its plates -- and where it
    ran); and the original topology algorithm's run time on it, measured or extrapolated
    (`kind`)."""
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
    frames: tuple[DemoFrame, ...]
    sheet: DemoSheet
    escape_scale: float
    machine: str
    topology: DemoTopologyExtrapolated | DemoTopologyMeasured


@dataclass(frozen=True)
class DemoNetwork:
    """One method's network on the compared block, drawn as the page's escape-time plate on that
    block's scale (`plate`, beside demo.json, `width` x `height` px), and its scores: the share of
    floor area it displaces, its L_P and its new road."""
    plate: str
    width: int
    height: int
    displacement: float
    permeability: float
    road_km: float


@dataclass(frozen=True)
class DemoComparison:
    """The methods compared on a block where topology has run: the block (id, buildings, place,
    Maps link, its rank under the shipped screen, its escape-time scale), the method's full
    build-out (`ours`, its run time and where it ran) and topology's network (`earlier`, its run
    time measured there)."""
    block_id: str
    n_buildings: int
    place: str
    maps_url: str
    proxy_rank: int
    escape_scale: float
    ours: DemoNetwork
    ours_seconds: float
    ours_machine: str
    earlier: DemoNetwork
    earlier_time: DemoTopologyMeasured


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
    """The method the page builds out: its reader-facing name, how it works, and the slider
    position the page opens on."""
    name: str
    blurb: str
    default_frame: int


@dataclass(frozen=True)
class DemoEncoding:
    """The plates' colours and the quantities their legend states. `escape_ramp` maps escape time
    from 0 to 100% of a block's `escape_scale`, short blue to long red, with an isochrone every
    `iso_step` of it. `road` is the grey of a new road's corridor and of a wholly destroyed
    building, which a building is once `gone_share` of it is gone; roads are `road_width_m` wide
    and conduct `road_conductance` times what open ground does. The block's edge: `street`
    (where people walk out), `edge` (a railway's or a ditch's line), `water` (a ditch).
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
    depth_ramp: tuple[str, ...]


@dataclass(frozen=True)
class DemoBundle:
    """The Demo page's numbers: the block it follows, the methods compared on another, the
    screens, the method, and the plates' encoding."""
    block: DemoBlock
    comparison: DemoComparison
    screen: DemoScreen
    method: DemoMethod
    encoding: DemoEncoding


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
            return f"{ts_type(args[0])}[]"
        return "[" + ", ".join(ts_type(a) for a in args) + "]"
    if typing.get_origin(t) is types.UnionType:
        return " | ".join("null" if a is type(None) else ts_type(a) for a in typing.get_args(t))
    if dataclasses.is_dataclass(t):
        return t.__name__
    raise TypeError(f"no TypeScript spelling for {t!r}")


def dts(top: type) -> str:
    """`top` and every dataclass it holds as TypeScript interfaces, each documented by its
    docstring, the held ones first."""
    order: list[type] = []

    def visit(cls: type) -> None:
        for hint in typing.get_type_hints(cls).values():
            for t in (hint, *typing.get_args(hint)):
                if dataclasses.is_dataclass(t) and t not in order:
                    visit(t)  # type: ignore[arg-type]
        if cls not in order:
            order.append(cls)

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
the roads one method builds through it, and a comparison of that method with the original topology
algorithm on block `{cid}`, where topology has run.

- `demo.json`: every number the page shows -- the block's place, the shipped screen's verdict on
  it (checked against `../screen-map/capetown.json`) and the two-stage depth_density screen's, the
  fires reported near it (each quoted from its report and checked against the article; none is
  placed inside the block, and the page draws none of them on a map), the method's frames under a
  displacement budget with its recorded run time, topology's run time (extrapolated there), and
  the comparison's scores and run times (topology's measured).
- `{sheet}`: the block's escape-time plates as one sprite sheet (the block with no new road
  first); `{ours}` and `{earlier}`: the two networks on `{cid}`, the same plate.

`web/src/demo.d.ts` is generated beside it from the same dataclasses, so a renamed field is a
TypeScript error.

The producer is `research/roadless/atlas/explore_proto/build.py` on the `research/roadless`
branch, which cannot merge into `main` (it carries a private submit group and a large GPU group):
the frames are the build-out pipeline's stored ones, so this directory is how they reach the
public site.
"""


def scaled(path: Path) -> Image.Image:
    im = Image.open(path)
    return im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)


def sprite(site: Site, fs: list[str], out: Path) -> tuple[int, int]:
    """The block's frames `fs` stacked at WIDTH px into `out`; (width, frame height)."""
    ims = [scaled(plate_name(site, f)) for f in fs]
    h = ims[0].height
    if any(i.height != h for i in ims):
        raise AssertionError(f"{site.bid}: frames of different heights")
    if h * len(ims) > 16383:
        raise SystemExit(f"{site.bid}: {len(ims)} frames of {h} px exceed WebP's 16383 px")
    sheet = Image.new("RGB", (WIDTH, h * len(ims)), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, (0, i * h))
    sheet.save(out, quality=QUALITY, method=6)
    return WIDTH, h


def single(src: Path, out: Path) -> tuple[int, int]:
    """One plate at WIDTH px into `out`; (width, height)."""
    im = scaled(src)
    im.convert("RGB").save(out, quality=QUALITY, method=6)
    return im.width, im.height


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


def frames_of(site: Site) -> tuple[list[DemoFrame], list[str]]:
    """The slider's frames on the block (the block with no new road first) and the sheet's frame
    names, in row order. Refuses a row with no network, or one where the run stopped short of
    the budget: the page shows a network that meets every budget."""
    pick = METHOD.pick(site.bid)
    key, src = pick.key(), pick.source(site)
    rows = src.rows(key)
    secs = METHOD.runtime.seconds(key, src, rows, site)
    fs = ["000"] + list(dict.fromkeys(r["f"] for r in rows if r is not None))
    frames = [DemoFrame(budget=0.0, displacement=0.0, road_km=0.0, permeability=0.0,
                        conductance_m2=0.0, seconds=0.0, row=0)]
    for t, r, s in zip(nw.SLIDER, rows, secs, strict=True):
        if r is None or s is None or r.get("full", False):
            raise SystemExit(f"{site.bid}: no network meeting the {t:g} budget")
        frames.append(DemoFrame(budget=t, displacement=r["D"], road_km=r["km"],
                                permeability=r["LP"], conductance_m2=float(r["C"]), seconds=s,
                                row=fs.index(r["f"])))
    return frames, fs


def proxy_of(sc: Screen, bid: str) -> DemoProxy:
    b = sc.bundle
    i = b["block_id"].index(bid)
    return DemoProxy(n=int(b["n"][i]), area_m2=float(b["area_m2"][i]),
                     perimeter_m=float(b["perimeter_m"][i]), score=float(sc.score[i]),
                     rank=int((sc.score > sc.score[i]).sum()) + 1,
                     informal=bool(b["informal"][i]))


def page_block(sc: Screen, pl: Peeled, out: Path) -> DemoBlock:
    """Everything the page shows about PAGE; its sprite sheet into `out`."""
    site = Site.of(PAGE.bid)
    proxy = proxy_of(sc, PAGE.bid)
    peel_score = pl.scores[PAGE.bid]
    rings = peel_score * proxy.area_m2 / proxy.n          # depth_density = peel depth x n/A
    # The bundle rounds areas (to 0.1 m²), so the quotient lands within a hair of a whole number;
    # another block's n or A would miss it by far more.
    if abs(rings - round(rings)) > 1e-3:
        raise AssertionError(f"{PAGE.bid}: depth_density {peel_score} over n/A is not a whole "
                             f"number of rings ({rings}); the bundle's n or A is not the screen's")
    frames, fs = frames_of(site)
    file = f"{METHOD.stem}_{PAGE.bid}.webp"
    width, height = sprite(site, fs, out / file)
    lat, lon = centroid_wgs84(PAGE.bid)
    return DemoBlock(
        block_id=PAGE.bid, n_buildings=proxy.n, place=PAGE.place, lat=lat, lon=lon,
        maps_url=maps_url(lat, lon, PAGE.zoom), plainly_informal=PAGE.plainly_informal,
        proxy=proxy,
        peel=DemoPeel(score=float(peel_score),
                      rank=1 + sum(1 for v in pl.scores.values() if v > peel_score),
                      rings=round(rings)),
        fires=FIRES, frames=tuple(frames),
        sheet=DemoSheet(file=file, width=width, frame_height=height, rows=len(fs)),
        escape_scale=round(world(site).scale, 1), machine=METHOD.runtime.machine,
        topology=PAGE.topology.of(site, proxy.n))


def comparison(sc: Screen, out: Path) -> DemoComparison:
    """The methods compared on COMPARED: the method's full build-out and topology's network,
    their plates into `out`."""
    site = Site.of(COMPARED.bid)
    proxy = proxy_of(sc, COMPARED.bid)
    frames, fs = frames_of(site)
    full = frames[-1]
    name = f"{METHOD.stem}_{COMPARED.bid}.webp"
    w, h = single(plate_name(site, fs[full.row]), out / name)
    ours = DemoNetwork(plate=name, width=w, height=h, displacement=full.displacement,
                       permeability=full.permeability, road_km=full.road_km)
    d, lp, km = COMPARED.topology.scores(site)
    name = f"topology_{COMPARED.bid}.webp"
    w, h = single(topology_plate(site), out / name)
    earlier = DemoNetwork(plate=name, width=w, height=h, displacement=d, permeability=lp,
                          road_km=round(km, 2))
    lat, lon = centroid_wgs84(COMPARED.bid)
    return DemoComparison(
        block_id=COMPARED.bid, n_buildings=proxy.n, place=COMPARED.place,
        maps_url=maps_url(lat, lon, COMPARED.zoom), proxy_rank=proxy.rank,
        escape_scale=round(world(site).scale, 1), ours=ours, ours_seconds=full.seconds,
        ours_machine=METHOD.runtime.machine, earlier=earlier,
        earlier_time=COMPARED.topology.of(site, proxy.n))


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
    block, compared = page_block(sc, pl, out), comparison(sc, out)
    floor = sc.floor
    depth = matplotlib.colormaps[reblock_render._CMAP]
    bundle = DemoBundle(
        block=block, comparison=compared,
        screen=DemoScreen(metric=SCREEN_METRIC, floor=float(floor["value"]), pool=int(floor["n"]),
                          precision=float(floor["precision"]), recall=float(floor["recall"]),
                          n_blocks=len(sc.score), peel_metric=PEEL_EXAMPLE,
                          peel_keep_n=pl.keep_n),
        method=DemoMethod(name=METHOD.name, blurb=METHOD.blurb, default_frame=DEFAULT),
        encoding=DemoEncoding(
            road=render.ROAD, street=rnd.STREET, edge=rnd.EDGE, water=nw.cp.gm.WATER_FILL,
            escape_ramp=tuple(mcolors.to_hex(SEQ(t)) for t in np.linspace(0, 1, 11)),
            iso_step=ISO_STEP, road_width_m=road_simp.ROAD_W,
            road_conductance=growth.ROAD_KAPPA, gone_share=GONE_SHARE,
            depth_ramp=tuple(mcolors.to_hex(depth(t)) for t in np.linspace(0, 1, 9))))
    (out / "demo.json").write_text(json.dumps(dataclasses.asdict(bundle), indent=1,
                                              ensure_ascii=False) + "\n")
    (out / "README.md").write_text(SITE_README.format(
        bid=block.block_id, cid=compared.block_id, sheet=block.sheet.file,
        ours=compared.ours.plate, earlier=compared.earlier.plate))
    (root / SITE_DTS).write_text(dts(DemoBundle))
    mb = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"{out}: {len(list(out.iterdir()))} files, {mb:.1f} MB; {root / SITE_DTS}", flush=True)
    print(f"  {block.block_id}: proxy #{block.proxy.rank} {block.proxy.score:.4f}; peel "
          f"#{block.peel.rank} {block.peel.score:.4f} ({block.peel.rings} rings); topology "
          f"{block.topology.kind}; compared on {compared.block_id}: ours D "
          f"{compared.ours.displacement} in {compared.ours_seconds} s, topology D "
          f"{compared.earlier.displacement:.3f} in {compared.earlier_time.seconds:.0f} s",
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
