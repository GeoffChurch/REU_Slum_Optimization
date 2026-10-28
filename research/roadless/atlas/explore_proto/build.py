"""A prototype of the public site's Explore page, Methods stage, redesigned (owner 2026-10-28): one
method's build-out on block 5810 under a budget slider in place of the GIF, with three stacked
plots whose dots follow the slider (permeability L_P against displacement, road length and
conductance added), the block's Google Maps link, the method's recorded run time beside an
extrapolation of the original topology algorithm's, and the plates re-coloured so that red is bad
and blue good. Before it (owner 2026-10-28, "we should have a screening section"): the Screening
stage, the site's Cape Town screen bundle (examples/screen-map/capetown.json) lightened into
screen.json (screen_bake) and drawn by the page with a floor slider, a city map and a close-up of
the block; then news reports of fires near the block (a list; the map marks none of them, since
some reports' streets cannot be placed). A claude.ai artifact at plates/artifact/ (the page and
its files), its publish list plates/artifact_publish.json; everything under plates/ is generated
(git-ignored).

Reuses the 5810 build-out page's pipeline (GeoffChurch/REU_Slum_Optimization's
research/roadless/atlas/roads5810/): the method's page rows from networks.py's sources
(GrownSource, family_of, Source.rows), the frames' stored rasters and escape times (networks.DATA,
written by compute.frames_of: no flow is recomputed here), render.py's plate helpers
(network_plate, the edge, `small`) and its building and road colours. Only the colour maps
differ, and they are this script's own (SEQ, SAVED below), so the build-out page's plates are
untouched.

The method is one setting (METHOD): a Shown, which names its stored frames (a Pick), the
reader-facing name and blurb, and how its run time is read (a Runtime Strategy: a grown run's
steps, or an optimizer's rows per budget). Swapping to the partition SIMP's cut is
`METHOD = PARTITION_CUT`, once networks.py `add` has written its frames.

Views (VIEWS): the roads (network_plate as on the build-out page); each home's escape time on one
fixed scale (the plain baseline's 99th percentile as 100%), long red and short blue, with
isochrones every ISO_STEP of that scale (the homes' escape times interpolated between their
centroids, smoothed, inside the block); and the share of each home's escape time saved over its
own mesh's baseline (0 to SAVED_MAX, sequential blue: owner 2026-10-28, the diverging map's red
half was almost unused, since a road only adds a way through).

The public site's Demo page (owner 2026-10-28, "make this a new 'demo' page on the website") is
this prototype ported onto main, which cannot merge this branch: `site <checkout>` writes what the
page needs into a main checkout as a committed bundle (SITE_DIR: demo.json, the three sprite
sheets, a README naming this script) and the bundle's TypeScript declaration (SITE_DTS, generated
from the bundle's dataclasses below, so a renamed field is a type error on main). Every number the
page shows is in demo.json, the fires' facts included (FIRES, quoted from their reports); the
site's generator reads it and the page's prose types none.

    PYTHONPATH=. uv run python research/roadless/atlas/explore_proto/build.py plates   # ~10 min
    PYTHONPATH=. uv run python research/roadless/atlas/explore_proto/build.py page
    PYTHONPATH=. uv run python research/roadless/atlas/explore_proto/build.py          # both
    PYTHONPATH=. uv run python research/roadless/atlas/explore_proto/build.py site ~/src/wt_demo
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import html as htmllib
import json
import shutil
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol

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
import networks as nw  # noqa: E402  (the build-out page's families; imports compute and render)
import pandas as pd  # noqa: E402
import render  # noqa: E402
import road_simp  # noqa: E402
import shapely  # noqa: E402
from PIL import Image  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

cp, rnd = nw.cp, render.rnd
PLATES = HERE / "plates"                 # git-ignored: everything this script writes
FRAMES = PLATES / "frames"               # full-size plates, one per frame and view
OUT = PLATES / "artifact"                # the artifact: index.html, s/, the figure
TEMPLATE = HERE / "page.html"            # the page, with this script's {{...}} fields
WIDTH = 800                              # px: a sprite frame's width (roads5810's artifact.WIDTH)
QUALITY = 80                             # lossy WebP's
BATCH_MB, BATCH_FILES = 60.0, 250        # a publish's files at most (roads5810's artifact.py)
VIEWS = ("network", "time", "saved")     # the plates' views, in the page's order


# ------------------------------------------------------------------ the method (one setting)

class Runtime(Protocol):
    """How long the method took to build each slider row's network, read from its own rows."""

    @property
    def machine(self) -> str:
        """Where it ran, in words (the rows do not record the host)."""
        ...

    def seconds(self, key: str, src: Any, rows: list[dict[str, Any] | None]) -> list[float | None]:
        """Per slider row the run time to that row's network (None where the row is none)."""
        ...


@dataclass(frozen=True)
class GrownSteps:
    """A growth or ears run (networks.GrownSource): steps.parquet's cumulative t at the row's
    stop, the state after its first m steps (the stop's prefix: networks.Grown)."""
    machine: str

    def seconds(self, key: str, src: Any, rows: list[dict[str, Any] | None]) -> list[float | None]:
        steps = pd.read_parquet(src.dir / "steps.parquet").set_index("step").t
        info = nw.info_of(key, "slider")
        m_of = {f: m for m, f in render.frame_names(info).items()}
        return [None if r is None else float(steps[m_of[r["f"]]]) for r in rows]


@dataclass(frozen=True)
class AnswerRows:
    """An optimizer's answer per budget (networks.AnswerSource over a Lined or Joint design): the
    run's own t_total (its rows' last), each row its budget's run's."""
    machine: str

    def seconds(self, key: str, src: Any, rows: list[dict[str, Any] | None]) -> list[float | None]:
        design = src.design
        return [None if r is None else
                float(nw.last_row(design.dir(r["b"]) / f"{nw.BID}.parquet").t_total) for r in rows]


class Pick(Protocol):
    """A method's stored frames on the build-out page: its key there (networks.DATA's file stem)
    and its networks.Source (runs and slider rows)."""

    def key(self) -> str: ...

    def source(self) -> Any: ...


@dataclass(frozen=True)
class GrownRun:
    """A growth or ears run by its plan (networks.GrownSource). Not through networks.Ears, which
    refuses a plan of an older ears revision: the build-out page's paths run is rev 1, and rev 2
    (ears.REV) changed only `divides`."""
    plan: str
    budget: float

    def key(self) -> str:
        return self.plan

    def source(self) -> Any:
        return nw.GrownSource(self.plan, budget=self.budget)


@dataclass(frozen=True)
class Family:
    """A networks.py family by its command line (networks.family_of)."""
    argv: tuple[str, ...]

    def key(self) -> str:
        return nw.family_of(list(self.argv)).entry().key

    def source(self) -> Any:
        return nw.family_of(list(self.argv)).source()


@dataclass(frozen=True)
class Shown:
    """The method the page shows: its stored frames, its reader-facing name, a sentence on how it
    works, a file stem, and how its run time is read."""
    pick: Pick
    name: str
    blurb: str
    stem: str
    runtime: Runtime


EARS_PATHS = Shown(
    pick=GrownRun("ears.paths.b20.gl0.25Croad.n2s0.1.M8.K4.D0.15.v1", budget=0.15),
    name="Loop roads",
    blurb=("Each new road runs from the street, or a road already built, across the block and "
           "back to a street or road, so every road is on a loop and none is a dead end. At each "
           "step the method tries the cheapest routes through the gaps between buildings and "
           "builds the one that most improves permeability for the homes it displaces."),
    stem="loop",
    runtime=GrownSteps(machine="one process on a workstation CPU (Intel Xeon w7-3455)"))

R3 = ("pt.K5.R10.band.t0.2,0.1,0.05,0.03,0.02.b1,2,4,8,16.q1,2,3,3,3.e0.2,0.1,0.05,0.02,0.i40")
PARTITION_CUT = Shown(
    pick=Family(("partition", "cut", f"{R3}.D{{D}}.s0.1.spec{{n}}s0.p1e0.0001v4")),
    name="Sub-block roads",
    blurb=("Divides the block into sub-blocks and lays a road wherever two sub-blocks meet, so "
           "every road is on a loop. The sub-blocks start from clusters of buildings and move "
           "until the roads between them most improve permeability within the budget; each "
           "budget is solved afresh."),
    stem="part",
    runtime=AnswerRows(machine="partition_rows' t_total (rows do not name the machine)"))

METHOD: Shown = EARS_PATHS


# ------------------------------------------------------------------ the topology extrapolation

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


class Extrapolation(NamedTuple):
    exponent: float          # the fit's: seconds ~ n^exponent
    se: float                # its standard error
    central: float           # seconds at n, by the fit
    low: float               # seconds at n: the smaller blocks' exponent from the largest run
    high: float              # seconds at n: the two largest runs' slope from the largest run
    small_exponent: float
    top_exponent: float
    largest: Timed
    n: int


def extrapolate(runs: tuple[Timed, ...], n: int) -> Extrapolation:
    """A least-squares power law through every run (log seconds on log n), and a range anchored on
    the largest run: the exponent of the runs below it (low) and the slope between the two
    largest (high). The runs steepen with size, so the fit through all of them under-predicts the
    largest; the range says how far the extrapolation could go either way."""
    x = np.log([r.n for r in runs])
    y = np.log([r.seconds for r in runs])
    (b, a), *_ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)
    resid = y - (a + b * x)
    se = float(np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum()))
    (bs, _as), *_ = np.linalg.lstsq(np.vstack([x[:-1], np.ones(len(x) - 1)]).T, y[:-1],
                                    rcond=None)
    top = runs[-1]
    bt = float((y[-1] - y[-2]) / (x[-1] - x[-2]))
    return Extrapolation(exponent=float(b), se=se, central=float(np.exp(a) * n ** b),
                         low=top.seconds * (n / top.n) ** bs, high=top.seconds * (n / top.n) ** bt,
                         small_exponent=float(bs), top_exponent=bt, largest=top, n=n)


def span(seconds: float) -> str:
    """A duration in the largest unit that keeps it readable."""
    for unit, s in (("centuries", 100 * 3.156e7), ("years", 3.156e7), ("months", 2.63e6),
                    ("days", 86400.0), ("hours", 3600.0), ("minutes", 60.0)):
        if seconds >= 1.5 * s:
            v = seconds / s
            return f"{v:.0f} {unit}" if v >= 10 else f"{v:.1f} {unit}"
    return f"{seconds:.0f} seconds"


# ------------------------------------------------------------------ colour maps (OKLab)

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
# one-signed (share of escape time saved): none a pale blue-grey (still visible on white),
# most a deep blue, lightness falling evenly
SAVED = ramp("saved_seq", [(0.92, 0.020, 250.0), (0.66, 0.100, 250.0), (0.38, 0.130, 262.0)])
SAVED_MAX = 0.75         # the scale's top: the homes' 1st percentile saves 72% at D 0.15
ISO_STEP = 0.10          # the isochrones' spacing, a share of the baseline's 99th percentile
ISO_GRID = 2.0           # m: the interpolated escape-time field's cell
ISO_SMOOTH = 4.0         # m: its Gaussian smoothing
DEFAULT = 5              # the slider's opening position (D 0.05)


# ------------------------------------------------------------------ plates

class World(NamedTuple):
    """What every plate shares: the frame, the buildings, the block's edge and polygon, the
    plain baseline's escape times and the escape-time scale."""
    fr: Any
    polys: np.ndarray
    centroids: np.ndarray
    edge: Any
    inside: Any
    u_plain: np.ndarray
    scale: float             # the plain baseline's 99th percentile: 100% on the time view


def world() -> World:
    b = np.load(cp.DATA / "baseline.npz", allow_pickle=True)          # our own file: WKB
    polys = shapely.from_wkb(b["polys"])
    [boundary] = shapely.from_wkb(b["boundary"])
    u = np.load(cp.DATA / "baseline_times.npz")["u"]
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


def frame_axes(w: World):  # type: ignore[no-untyped-def]
    f = w.fr.frame
    fig = plt.figure(figsize=(f.nx / 100, f.ny / 100), dpi=100)
    return fig, fig.add_axes((0, 0, 1, 1))


def finish(fig, ax, w: World, out: Path) -> None:  # type: ignore[no-untyped-def]
    render.draw_edge(ax, w.fr, w.edge)
    ax.set_xlim(w.fr.lim[0], w.fr.lim[1])
    ax.set_ylim(w.fr.lim[2], w.fr.lim[3])
    ax.set_axis_off()
    tmp = render.scratch(out)
    fig.savefig(tmp, dpi=100, facecolor="white")
    plt.close(fig)
    render.small(tmp, out)


def homes_plate(w: World, value: np.ndarray, *, cmap, norm, gone: np.ndarray, corridor,  # type: ignore[no-untyped-def]
                iso: tuple[np.ndarray, np.ndarray, np.ndarray] | None, out: Path) -> None:
    """render.time_plate's figure on this script's map: each home filled by `value` (nan: no
    home, outlined), the gone buildings and the corridor in render.ROAD, and with `iso` the
    isochrones (every ISO_STEP of the scale, dark lines on a white halo)."""
    fig, ax = frame_axes(w)
    has = np.isfinite(value) & ~gone
    gpd.GeoSeries(list(w.polys[has])).plot(ax=ax, color=cmap(norm(value[has])), lw=0)
    gpd.GeoSeries(list(w.polys[~has & ~gone])).boundary.plot(ax=ax, color=rnd.BUILDING, lw=0.4)
    gpd.GeoSeries(list(w.polys[gone])).plot(ax=ax, color=render.ROAD, lw=0)
    if corridor is not None:
        gpd.GeoSeries([corridor]).plot(ax=ax, color=render.ROAD, lw=0)
    if iso is not None:
        X, Y, Z = iso
        levels = np.arange(ISO_STEP, 1.0 + 1e-9, ISO_STEP) * w.scale
        cs = ax.contour(X, Y, Z, levels=levels, colors=rnd.STREET, linewidths=1.1)
        cs.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white", alpha=0.85),
                             pe.Normal()])
    finish(fig, ax, w, out)


def saved(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    """The share of each home's escape time saved, 1 - after / before, from 0 (a home a little
    slower on the carved mesh, a few dozen at D 0.15, shown as none saved) to SAVED_MAX; a home
    freed (no time before) at SAVED_MAX, nan with none after."""
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(np.isfinite(before) & (before > 0), 1.0 - after / before, SAVED_MAX)
    return np.where(np.isfinite(after), np.clip(g, 0.0, SAVED_MAX), np.nan)


def plate_name(f: str, view: str) -> Path:
    return FRAMES / f"{METHOD.stem}_{f}_{view}.webp"


def plates() -> None:
    """Every frame of the method's runs, and the baseline, in VIEWS, into FRAMES."""
    FRAMES.mkdir(parents=True, exist_ok=True)
    w = world()
    tn = mcolors.Normalize(0.0, w.scale, clip=True)
    sn = mcolors.Normalize(0.0, SAVED_MAX, clip=True)
    none = np.zeros(len(w.polys), dtype=bool)
    render.network_plate(w.fr, w.polys, np.zeros(len(w.polys)), None, edge=w.edge,
                         out=plate_name("000", "network"))
    ok = np.isfinite(w.u_plain)
    homes_plate(w, w.u_plain, cmap=SEQ, norm=tn, gone=none, corridor=None,
                iso=field(w, w.u_plain, ok), out=plate_name("000", "time"))
    homes_plate(w, np.where(ok, 0.0, np.nan), cmap=SAVED, norm=sn, gone=none, corridor=None,
                iso=None, out=plate_name("000", "saved"))
    key, src = METHOD.pick.key(), METHOD.pick.source()
    for run in src.runs():
        info = nw.info_of(key, run.name)
        d = np.load(nw.DATA / f"{key}_{run.name}.npz", allow_pickle=True)      # our own: WKB
        t = np.load(nw.DATA / f"{key}_{run.name}_times.npz")
        if not np.allclose(d["frame"], np.asarray(w.fr.frame, dtype=float)):
            raise AssertionError(f"{key} {run.name}: its frame is not the baseline's")
        names = render.frame_names(info)
        for mm in [int(s) for s in d["stops"]]:
            f = names[mm]
            [corridor] = shapely.from_wkb(d[f"corridor_{mm}"])
            c = d[f"c_{mm}"]
            gone = c >= 1.0
            u = t[f"u_{mm}"]
            render.network_plate(w.fr, w.polys, c, corridor, edge=w.edge,
                                 out=plate_name(f, "network"))
            homes_plate(w, u, cmap=SEQ, norm=tn, gone=gone, corridor=corridor,
                        iso=field(w, u, np.isfinite(u) & ~gone), out=plate_name(f, "time"))
            homes_plate(w, saved(u, t["u_base"]), cmap=SAVED, norm=sn, gone=gone,
                        corridor=corridor, iso=None, out=plate_name(f, "saved"))


# ------------------------------------------------------------------ the screen

SCREEN = cp.ROOT / "examples" / "screen-map" / "capetown.json"   # the site's ScreenMap bundle
SCREEN_METRIC = "depth_density_proxy"   # the shipped screen (its floor read from the bundle)
CITY_TOL = 30.0          # m: the city map's simplification (a pixel is ~90 m at 800 px)
LOCAL_TOL = 2.0          # m: the close-up's
LOCAL_PAD = 450.0        # m: the close-up's margin around the block


def deltas(ring: np.ndarray) -> list[int]:
    """A ring as integer metres, its first point absolute and each next one the step from the
    last (x, y interleaved): small numbers, a small file."""
    c = np.round(ring).astype(np.int64)
    return np.concatenate([c[:1], np.diff(c, axis=0)]).ravel().tolist()


def simplified(rings: list[list[list[float]]], tol: float) -> list[list[int]]:
    """A block's rings (even-odd, the bundle's) simplified by `tol`, a ring that collapses
    dropped (the first kept as it is, so no block vanishes)."""
    out: list[list[int]] = []
    for ring in rings:
        g = shapely.Polygon(ring).simplify(tol, preserve_topology=False)
        if g.is_empty or g.geom_type != "Polygon" or g.area < tol * tol:
            if not out:
                out.append(deltas(np.asarray(ring)[:-1]))
            continue
        out.append(deltas(np.asarray(g.exterior.coords)[:-1]))
    return out


class Screen(NamedTuple):
    """The site's Cape Town screen bundle (its parsed JSON), every block's score under the shipped
    screen (gen_screen_map._score, the widget's own arithmetic), the shipped floor's entry and the
    followed block's index."""
    bundle: dict[str, Any]
    score: np.ndarray
    floor: dict[str, Any]
    follow: int


def screen() -> Screen:
    """SCREEN under SCREEN_METRIC; refuses a bundle that follows another block than the page's."""
    from scripts.gen_screen_map import _score  # noqa: PLC0415  (the repo's scripts package)

    b = json.loads(SCREEN.read_text())
    n, a, per = (np.asarray(b[k], dtype=float) for k in ("n", "area_m2", "perimeter_m"))
    [floor] = [f for f in b["floors"] if f["metric"] == SCREEN_METRIC]
    follow = b["follow"]["index"]
    if b["block_id"][follow] != cp.BID:
        raise SystemExit(f"the bundle follows {b['block_id'][follow]}, the page {cp.BID}")
    return Screen(bundle=b, score=_score(SCREEN_METRIC, n, a, per), floor=floor, follow=follow)


class ScreenFacts(NamedTuple):
    """The followed block under the shipped screen (its score and rank among the bundle's
    blocks), and the screen at its floor: the pool (the bundle's own count, checked), its
    precision and its recall against the City's survey."""
    metric: str
    score: float
    rank: int
    n_blocks: int
    floor: float
    pool: int
    precision: float
    recall: float


def screen_facts(sc: Screen) -> ScreenFacts:
    pool = int((sc.score >= sc.floor["value"]).sum())
    if pool != sc.floor["n"]:
        raise AssertionError("the screen's pool at its floor is not the bundle's")
    return ScreenFacts(metric=SCREEN_METRIC, score=float(sc.score[sc.follow]),
                       rank=int((sc.score > sc.score[sc.follow]).sum()) + 1,
                       n_blocks=len(sc.score), floor=float(sc.floor["value"]), pool=pool,
                       precision=float(sc.floor["precision"]), recall=float(sc.floor["recall"]))


def screen_bake(sc: Screen, out: Path) -> None:
    """The screen bundle, lightened for the prototype into `out`: each block's score, the City's
    informal survey flag, rings simplified for the city map and, inside the close-up's window,
    for the close-up; the shipped floor, the followed block (local metres, the bundle's
    origin)."""
    b, floor, follow = sc.bundle, sc.floor, sc.follow
    a = np.asarray(b["area_m2"], dtype=float)
    mine = np.asarray(b["rings"][follow][0])
    xs, ys = mine[:, 0], mine[:, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(np.ptp(xs), np.ptp(ys)) / 2 + LOCAL_PAD
    box = shapely.box(cx - half, cy - half, cx + half, cy + half)
    near = [i for i, r in enumerate(b["rings"]) if shapely.Polygon(r[0]).intersects(box)]
    order = np.argsort(-a, kind="stable")                  # the large first, the small over them
    data = dict(
        metric=SCREEN_METRIC, floor=floor["value"], follow=follow,
        s=[float(f"{v:.7g}") for v in sc.score], inf="".join(str(int(v)) for v in b["informal"]),
        order=order.tolist(), city=[simplified(r, CITY_TOL) for r in b["rings"]],
        box=[round(cx - half), round(cy - half), round(cx + half), round(cy + half)],
        near=near, local=[simplified(b["rings"][i], LOCAL_TOL) for i in near])
    if sum(v >= floor["value"] for v in data["s"]) != floor["n"]:
        raise AssertionError("the rounded scores' pool at the floor is not the bundle's")
    out.write_text(json.dumps(data, separators=(",", ":")))


# ------------------------------------------------------------------ the place and the fires

PLACE = "the Bloekombos and Wallacedene area of Kraaifontein"     # the block's, in words


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
    """One fire reported near the block: its month (ISO, YYYY-MM), where the reports place it
    (with OpenStreetMap's distance to the block where the street can be found), what they say
    happened, and the reports. Each was checked against its article (2026-10-28); none is placed
    inside the block, and the page draws none of them on a map (owner 2026-10-28: some reports'
    streets cannot be placed)."""
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


def fires_html() -> str:
    """FIRES as the prototype's list."""
    def report(r: Report) -> str:
        d = dt.date.fromisoformat(r.date)
        title = "" if r.title is None else f": “{htmllib.escape(r.title, quote=False)}”"
        return (f'<a href="{htmllib.escape(r.url)}" target="_blank" rel="noopener">'
                f"{htmllib.escape(r.outlet, quote=False)}, {d.day} {d:%B %Y}{title} ↗</a>")

    items = "".join(
        f"      <li>\n"
        f'        <div class="when">{dt.date.fromisoformat(f.month + "-01"):%b %Y}</div>\n'
        f'        <div class="what">\n'
        f'          <p class="where">{htmllib.escape(f.where, quote=False)}</p>\n'
        f"          <p>{htmllib.escape(f.what, quote=False)}</p>\n"
        f'          <p class="src">{" · ".join(report(r) for r in f.reports)}</p>\n'
        f"        </div>\n"
        f"      </li>\n" for f in FIRES)
    return f'    <ol class="fires">\n{items}    </ol>'


# ------------------------------------------------------------------ the page

def scaled(path: Path) -> Image.Image:
    im = Image.open(path)
    return im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)


def sprite(fs: list[str], view: str, out: Path) -> tuple[int, int]:
    """The frames `fs` of `view` stacked at WIDTH px into `out`; (width, frame height)."""
    ims = [scaled(plate_name(f, view)) for f in fs]
    h = ims[0].height
    if any(i.height != h for i in ims):
        raise AssertionError(f"{view}: frames of different heights")
    if h * len(ims) > 16383:
        raise SystemExit(f"{view}: {len(ims)} frames of {h} px exceed WebP's 16383 px")
    sheet = Image.new("RGB", (WIDTH, h * len(ims)), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, (0, i * h))
    sheet.save(out, quality=QUALITY, method=6)
    return WIDTH, h


def stops_css(cmap, n: int = 11) -> str:  # type: ignore[no-untyped-def]
    """A CSS linear-gradient of `cmap` (the legend's bar, drawn by the page on white)."""
    return ", ".join(f"{mcolors.to_hex(cmap(t))} {t * 100:.0f}%" for t in np.linspace(0, 1, n))


def centroid_wgs84() -> tuple[float, float]:
    """The block's centroid (its own polygon, in its CRS) as (lat, lon)."""
    [b] = cp.common.build_blocks([cp.BID])
    c = shapely.centroid(b.boundary)
    lon, lat = Transformer.from_crs(b.crs, "EPSG:4326", always_xy=True).transform(c.x, c.y)
    return float(lat), float(lon)


def fill(template: str, fields: dict[str, str]) -> str:
    """`template` with each {{name}} replaced; refuses a field missing either way."""
    out = template
    for k, v in fields.items():
        tok = "{{" + k + "}}"
        if tok not in out:
            raise SystemExit(f"page.html: no {tok}")
        out = out.replace(tok, v)
    if "{{" in out:
        raise SystemExit(f"page.html: an unfilled field near {out[out.index('{{'):][:40]!r}")
    return out


def maps_url(lat: float, lon: float) -> str:
    """Google Maps' satellite view centred on (lat, lon)."""
    return (f"https://www.google.com/maps/@?api=1&map_action=map&center={lat:.5f}%2C{lon:.5f}"
            f"&zoom=17&basemap=satellite")


class Built(NamedTuple):
    """What both pages say about the method on the block: the slider rows (render.SLIDER's) and
    the run time to each, the sprite sheets' frames (`fs`, "000" the no-road baseline first), the
    block's centroid, its homes, the method's whole run time, the topology extrapolation at the
    block's size and the escape-time view's scale (World.scale)."""
    rows: list[dict[str, Any] | None]
    secs: list[float | None]
    fs: list[str]
    lat: float
    lon: float
    n_homes: int
    total: float
    ex: Extrapolation
    scale: float


def built() -> Built:
    key, src = METHOD.pick.key(), METHOD.pick.source()
    rows = src.rows(key)
    secs = METHOD.runtime.seconds(key, src, rows)
    lat, lon = centroid_wgs84()
    n_homes = len(np.load(cp.DATA / "baseline_times.npz")["u"])
    return Built(rows=rows, secs=secs,
                 fs=["000"] + list(dict.fromkeys(r["f"] for r in rows if r is not None)),
                 lat=lat, lon=lon, n_homes=n_homes, total=max(s for s in secs if s is not None),
                 ex=extrapolate(TOPOLOGY_RUNS, n_homes), scale=world().scale)


def page() -> None:
    b = built()
    rows, secs, fs, ex, total = b.rows, b.secs, b.fs, b.ex, b.total
    lat, lon, n_homes = b.lat, b.lon, b.n_homes
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "s").mkdir(parents=True)
    size = None
    for v in VIEWS:
        size = sprite(fs, v, OUT / "s" / f"{METHOD.stem}_{v}.webp")
    assert size is not None
    frames = [dict(i=0, target=0.0, D=0.0, km=0.0, LP=0.0, C=0, t=0.0, none=False, held=False)]
    for k, (r, s) in enumerate(zip(rows, secs, strict=True), start=1):
        if r is None:
            frames.append(dict(i=0, target=k / 100, D=0.0, km=0.0, LP=0.0, C=0, t=None,
                               none=True, held=False))
        else:
            frames.append(dict(i=fs.index(r["f"]), target=k / 100, D=r["D"], km=r["km"],
                               LP=r["LP"], C=r["C"], t=s, none=False,
                               held=bool(r.get("held", False)), full=bool(r.get("full", False))))
    scr = screen()
    screen_bake(scr, OUT / "screen.json")
    sc = screen_facts(scr)
    data = dict(frames=frames, sheet=dict(n=len(fs), w=size[0], h=size[1]),
                stem=METHOD.stem, default=DEFAULT, iso=ISO_STEP)
    runs = "; ".join(f"{r.n:,} buildings {r.seconds:,.1f} s" for r in TOPOLOGY_RUNS)
    fields = {
        "NAME": htmllib.escape(METHOD.name),
        "BLURB": htmllib.escape(METHOD.blurb),
        "BID": cp.BID,
        "PLACE": htmllib.escape(PLACE),
        "LAT": f"{abs(lat):.4f}° {'S' if lat < 0 else 'N'}",
        "LON": f"{abs(lon):.4f}° {'E' if lon >= 0 else 'W'}",
        "MAPS": htmllib.escape(maps_url(lat, lon)),
        "HOMES": f"{n_homes:,}",
        "OURS": span(total),
        "OURS_MACHINE": htmllib.escape(METHOD.runtime.machine),
        "TOPO": span(ex.central),
        "TOPO_LOW": span(ex.low),
        "TOPO_HIGH": span(ex.high),
        "TOPO_EXP": f"{ex.exponent:.1f}",
        "TOPO_SE": f"{ex.se:.1f}",
        "TOPO_SMALL_EXP": f"{ex.small_exponent:.1f}",
        "TOPO_TOP_EXP": f"{ex.top_exponent:.1f}",
        "TOPO_LARGEST_N": f"{ex.largest.n:,}",
        "TOPO_LARGEST_H": f"{ex.largest.seconds / 3600:.1f}",
        "TOPO_RUNS": runs,
        "TOPO_RATIO": f"{float(f'{ex.central / total:.2g}'):,.0f}",   # two figures
        "SEQ": stops_css(SEQ),
        "SEQ_HEX": json.dumps([mcolors.to_hex(SEQ(t)) for t in np.linspace(0, 1, 11)]),
        "SAVED": stops_css(SAVED),
        "ISO_PCT": f"{ISO_STEP * 100:.0f}",
        "SAVED_TICKS": "".join(f"<span>{t}</span>" for t in
                               ["none"] + [f"{SAVED_MAX * 100 * k / 3:.0f}%" for k in (1, 2)]
                               + [f"{SAVED_MAX * 100:.0f}%+"]),
        "SCALE": f"{b.scale:.0f}",
        "STEM": METHOD.stem,
        "DATA": json.dumps(data, separators=(",", ":")),
        "FIRES": fires_html(),
        "SC_SCORE": f"{sc.score:.4f}",
        "SC_FLOOR": f"{sc.floor:g}",
        "SC_TIMES": f"{sc.score / sc.floor:.1f}",
        "SC_RANK": f"{sc.rank:,}",
        "SC_N": f"{sc.n_blocks:,}",
        "SC_TOP": f"{100 * sc.rank / sc.n_blocks:.2g}",
        "SC_POOL": f"{sc.pool:,}",
        "SC_PREC": f"{100 * sc.precision:.0f}",
        "SC_REC": f"{100 * sc.recall:.0f}",
    }
    html = fill(TEMPLATE.read_text(), fields)
    (OUT / "index.html").write_text(html)
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, '
            'viewport-fit=cover">\n<base href="artifact/">\n<style>body{margin:0;'
            'font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}'
            '</style>\n</head>\n<body>\n')
    (PLATES / "preview.html").write_text(head + html + "\n</body>\n</html>\n")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p != OUT / "index.html")
    batches: list[dict[str, str]] = [{}]
    mb = 0.0
    for f in files:
        fmb = f.stat().st_size / 1e6
        if batches[-1] and (mb + fmb > BATCH_MB or len(batches[-1]) >= BATCH_FILES):
            batches.append({})
            mb = 0.0
        batches[-1][str(f.relative_to(OUT))] = str(f)
        mb += fmb
    (PLATES / "artifact_publish.json").write_text(json.dumps(
        dict(page=str(OUT / "index.html"), batches=batches), indent=1))
    allmb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"{len(files) + 1} files, {allmb:.1f} MB in {OUT}; topology {span(ex.central)} "
          f"(n^{ex.exponent:.2f} +- {ex.se:.2f}; {span(ex.low)} -- {span(ex.high)}); ours "
          f"{total:.0f} s; centre {lat:.5f}, {lon:.5f}", flush=True)


# ------------------------------------------------------------------ the site's bundle

SITE_DIR = Path("examples") / "demo"            # in a main checkout: the bundle
SITE_DTS = Path("web") / "src" / "demo.d.ts"   # in a main checkout: the bundle's declaration
# A building counts as wholly destroyed once this share of it is gone (or a road cuts it in two):
# the displacement rule the build-out page's network plate states (roads5810's caption).
GONE_SHARE = 0.1


@dataclass(frozen=True)
class DemoScreen:
    """The block under the shipped screen (ScreenFacts), and the close-up's margin around it (m):
    the site's screen map frames its close-up of the block the way the prototype's does."""
    metric: str
    score: float
    rank: int
    n_blocks: int
    floor: float
    pool: int
    precision: float
    recall: float
    closeup_pad_m: float


@dataclass(frozen=True)
class DemoFrame:
    """One slider position: the network the method has built by the displacement budget `budget`
    (a share of the homes' floor area; 0 the block with no new road). `displacement` is the share
    it reaches (at least `budget`), `road_km` its new road, `permeability` its L_P,
    `conductance_m2` the conductance it adds (m² of open ground), `seconds` the method's run time
    to it, `row` its frame's row in each sprite sheet."""
    budget: float
    displacement: float
    road_km: float
    permeability: float
    conductance_m2: float
    seconds: float
    row: int


@dataclass(frozen=True)
class DemoMethod:
    """The method the page builds out: its reader-facing name, how it works, where its run time
    was recorded, the slider's frames (`default_frame` the one the page opens on)."""
    name: str
    blurb: str
    machine: str
    default_frame: int
    frames: tuple[DemoFrame, ...]


@dataclass(frozen=True)
class DemoSheets:
    """The plates as sprite sheets beside demo.json, one per view, each `rows` frames of `width` x
    `frame_height` px stacked top to bottom: the roads (`network`), each home's escape time
    (`time`), and the share of it the roads save (`saved`)."""
    width: int
    frame_height: int
    rows: int
    network: str
    time: str
    saved: str


@dataclass(frozen=True)
class DemoEncoding:
    """The plates' colours and the quantities their legends state. `escape_ramp` maps escape time
    from 0 to 100% of `escape_scale` (the model's units: the 99th-percentile home with no new
    road), short blue to long red, with an isochrone every `iso_step` of it; `saved_ramp` maps the
    share saved from none to `saved_max`. Roads are `road_width_m` wide and conduct
    `road_conductance` times what open ground does; a building counts as wholly destroyed once
    `gone_share` of it is gone. The block's edge: `street` (where people walk out), `edge` (a
    railway's or a ditch's line), `water` (a ditch)."""
    building: str
    destroyed: str
    road: str
    street: str
    edge: str
    water: str
    escape_ramp: tuple[str, ...]
    saved_ramp: tuple[str, ...]
    escape_scale: float
    iso_step: float
    saved_max: float
    road_width_m: float
    road_conductance: float
    gone_share: float


@dataclass(frozen=True)
class DemoTopology:
    """The original topology algorithm's run time on this block, extrapolated from `runs` (each
    once, at `alpha`, on one thread) to `n_buildings`: a least-squares power law through all of
    them (seconds ~ n^`exponent`, standard error `se`) gives `central_s`; anchored on the largest
    run, the smaller runs' exponent (`small_exponent`) gives `low_s` and the two largest runs'
    slope (`top_exponent`) `high_s`."""
    alpha: int
    runs: tuple[Timed, ...]
    n_buildings: int
    exponent: float
    se: float
    central_s: float
    low_s: float
    high_s: float
    small_exponent: float
    top_exponent: float


@dataclass(frozen=True)
class DemoBundle:
    """The Demo page's numbers: the block (its id, buildings, place in words, centroid and a Google
    Maps link), the screen's verdict on it, the fires reported near it, the method built out on it
    and its plates, and the topology algorithm's extrapolated run time on it."""
    block_id: str
    n_buildings: int
    place: str
    lat: float
    lon: float
    maps_url: str
    screen: DemoScreen
    fires: tuple[Fire, ...]
    method: DemoMethod
    sheets: DemoSheets
    encoding: DemoEncoding
    topology: DemoTopology


def ts_type(t: Any) -> str:
    """`t`'s TypeScript spelling, for the bundle's types only (refusing anything else)."""
    if t is str:
        return "string"
    if t is int or t is float:
        return "number"
    if typing.get_origin(t) is tuple:
        item, more = typing.get_args(t)
        if more is not Ellipsis:
            raise TypeError(f"{t!r}: only tuple[X, ...] has a TypeScript spelling here")
        return f"{ts_type(item)}[]"
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

What the site's Demo page reads: one Cape Town block, `{bid}`, followed from the city-wide screen
to the roads one method builds through it.

- `demo.json`: every number the page shows. The block and its place, the shipped screen's verdict
  on it (checked against `../screen-map/capetown.json`), the fires reported near it (each quoted
  from its report and checked against the article; none is placed inside the block, and the page
  draws none of them on a map), the method's frames under a displacement budget with its recorded
  run time, the plates' colours, and the original topology algorithm's run time on this block,
  extrapolated from {n_runs} recorded runs on smaller blocks.
- {sheets}: the method's plates, one sprite sheet per view ({rows} frames each, the block with no
  new road first).

`web/src/demo.d.ts` is generated beside it from the same dataclasses, so a renamed field is a
TypeScript error.

The producer is `research/roadless/atlas/explore_proto/build.py` on the `research/roadless`
branch, which cannot merge into `main` (it carries a private submit group and a large GPU group):
the frames are the 5810 build-out pilot's stored ones (`{plan}`), so this directory is how they
reach the public site.
"""


def site(root: Path) -> None:
    """The Demo page's bundle (SITE_DIR) and its declaration (SITE_DTS) into the main checkout
    `root`."""
    if not (root / "mkdocs.yml").is_file() or not (root / "web" / "src").is_dir():
        raise SystemExit(f"{root} is not a checkout of the site (no mkdocs.yml or web/src)")
    b = built()
    if any(r is None or s is None for r, s in zip(b.rows, b.secs, strict=True)):
        raise SystemExit("a slider row with no network: the site's widget draws a network at "
                         "every budget")
    out = root / SITE_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names = {v: f"{METHOD.stem}_{v}.webp" for v in VIEWS}
    size = None
    for v in VIEWS:
        size = sprite(b.fs, v, out / names[v])
    assert size is not None
    frames = [DemoFrame(budget=0.0, displacement=0.0, road_km=0.0, permeability=0.0,
                        conductance_m2=0.0, seconds=0.0, row=0)]
    for t, r, s in zip(render.SLIDER, b.rows, b.secs, strict=True):
        assert r is not None and s is not None                   # refused above
        frames.append(DemoFrame(budget=t, displacement=r["D"], road_km=r["km"],
                                permeability=r["LP"], conductance_m2=float(r["C"]), seconds=s,
                                row=b.fs.index(r["f"])))
    sc = screen_facts(screen())
    ex = b.ex
    bundle = DemoBundle(
        block_id=cp.BID, n_buildings=b.n_homes, place=PLACE, lat=b.lat, lon=b.lon,
        maps_url=maps_url(b.lat, b.lon),
        screen=DemoScreen(**sc._asdict(), closeup_pad_m=LOCAL_PAD),
        fires=FIRES,
        method=DemoMethod(name=METHOD.name, blurb=METHOD.blurb, machine=METHOD.runtime.machine,
                          default_frame=DEFAULT, frames=tuple(frames)),
        sheets=DemoSheets(width=size[0], frame_height=size[1], rows=len(b.fs),
                          network=names["network"], time=names["time"], saved=names["saved"]),
        encoding=DemoEncoding(
            building=rnd.BUILDING, destroyed=rnd.CLEARED, road=render.ROAD, street=rnd.STREET,
            edge=rnd.EDGE, water=nw.cp.gm.WATER_FILL,
            escape_ramp=tuple(mcolors.to_hex(SEQ(t)) for t in np.linspace(0, 1, 11)),
            saved_ramp=tuple(mcolors.to_hex(SAVED(t)) for t in np.linspace(0, 1, 11)),
            escape_scale=round(b.scale, 1), iso_step=ISO_STEP, saved_max=SAVED_MAX,
            road_width_m=road_simp.ROAD_W, road_conductance=growth.ROAD_KAPPA,
            gone_share=GONE_SHARE),
        topology=DemoTopology(
            alpha=TOPOLOGY_ALPHA, runs=TOPOLOGY_RUNS, n_buildings=ex.n, exponent=ex.exponent,
            se=ex.se, central_s=ex.central, low_s=ex.low, high_s=ex.high,
            small_exponent=ex.small_exponent, top_exponent=ex.top_exponent))
    (out / "demo.json").write_text(json.dumps(dataclasses.asdict(bundle), indent=1,
                                              ensure_ascii=False) + "\n")
    (out / "README.md").write_text(SITE_README.format(
        bid=cp.BID, n_runs=len(TOPOLOGY_RUNS), rows=len(b.fs), plan=METHOD.pick.key(),
        sheets=", ".join(f"`{n}`" for n in names.values())))
    (root / SITE_DTS).write_text(dts(DemoBundle))
    mb = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"{out}: {len(list(out.iterdir()))} files, {mb:.1f} MB; {root / SITE_DTS}", flush=True)


def main(argv: list[str]) -> None:
    matplotlib.use("Agg")
    match argv:
        case ["plates"]:
            plates()
        case ["page"]:
            page()
        case ["site", root]:
            site(Path(root).expanduser().resolve())
        case []:
            plates()
            page()
        case _:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
