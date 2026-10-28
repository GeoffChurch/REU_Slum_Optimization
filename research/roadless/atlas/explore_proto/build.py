"""A prototype of the public site's Explore page, Methods stage, redesigned (owner 2026-10-28): one
method's build-out on block 5810 under a budget slider in place of the GIF, with three stacked
plots whose dots follow the slider (permeability L_P against displacement, road length and
conductance added), the block's Google Maps link, the method's recorded run time beside an
extrapolation of the original topology algorithm's, and the plates re-coloured so that red is bad
and blue good. Before it (owner 2026-10-28, "we should have a screening section"): the Screening
stage, the site's Cape Town screen bundle (examples/screen-map/capetown.json) lightened into
screen.json (screen_bake) and drawn by the page with a floor slider, a city map and a close-up of
the block; then news reports of fires near the block, their streets placed by OpenStreetMap
(FIRES). A claude.ai artifact at plates/artifact/ (the page and its files), its publish list
plates/artifact_publish.json; everything under plates/ is generated (git-ignored).

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

    PYTHONPATH=. uv run python research/roadless/atlas/explore_proto/build.py plates   # ~10 min
    PYTHONPATH=. uv run python research/roadless/atlas/explore_proto/build.py page
    PYTHONPATH=. uv run python research/roadless/atlas/explore_proto/build.py          # both
"""
from __future__ import annotations

import html as htmllib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "roads5810"))
import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networks as nw  # noqa: E402  (the build-out page's families; imports compute and render)
import pandas as pd  # noqa: E402
import render  # noqa: E402
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

class Timed(NamedTuple):
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
LOCAL_PAD = 450.0        # m: the close-up's margin around the block and the fire markers


class Placed(NamedTuple):
    """A fire report's street as OpenStreetMap places it (Nominatim, 2026-10-28): a point on the
    named street, not the fire's own spot. `label` is the marker's text on the close-up."""
    label: str
    lat: float
    lon: float


# the reports whose street OSM names (the page's list says which could not be placed)
FIRES = (Placed("2016", -33.8399851, 18.7460948),      # Dlayedwa Street, Kraaifontein East
         Placed("2022", -33.8576754, 18.7369617),      # Pietersen Road, Kraaifontein
         Placed("2025", -33.8515658, 18.7417907))      # Maroela Road, Wallacedene


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


def screen_bake(out: Path) -> dict[str, Any]:
    """The site's Cape Town screen bundle, lightened for the page: each block's score under the
    shipped screen (gen_screen_map._score, the widget's own arithmetic), the City's informal
    survey flag, rings simplified for the city map and, inside the close-up's window, for the
    close-up; the shipped floor, the followed block, the fire markers (local metres, the bundle's
    origin). Written to `out`; returns the page's facts about the followed block."""
    from scripts.gen_screen_map import _score  # noqa: PLC0415  (the repo's scripts package)

    b = json.loads(SCREEN.read_text())
    n, a, per = (np.asarray(b[k], dtype=float) for k in ("n", "area_m2", "perimeter_m"))
    score = _score(SCREEN_METRIC, n, a, per)
    [floor] = [f for f in b["floors"] if f["metric"] == SCREEN_METRIC]
    follow = b["follow"]["index"]
    if b["block_id"][follow] != cp.BID:
        raise SystemExit(f"the bundle follows {b['block_id'][follow]}, the page {cp.BID}")
    ox, oy = b["origin"]
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{b['crs_epsg']}", always_xy=True)
    fires = []
    for f in FIRES:
        x, y = to_utm.transform(f.lon, f.lat)
        fires.append(dict(label=f.label, x=round(x - ox), y=round(y - oy)))
    mine = np.asarray(b["rings"][follow][0])
    xs = np.concatenate([mine[:, 0], [f["x"] for f in fires]])
    ys = np.concatenate([mine[:, 1], [f["y"] for f in fires]])
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(np.ptp(xs), np.ptp(ys)) / 2 + LOCAL_PAD
    box = shapely.box(cx - half, cy - half, cx + half, cy + half)
    near = [i for i, r in enumerate(b["rings"]) if shapely.Polygon(r[0]).intersects(box)]
    order = np.argsort(-a, kind="stable")                  # the large first, the small over them
    data = dict(
        metric=SCREEN_METRIC, floor=floor["value"], follow=follow,
        s=[float(f"{v:.7g}") for v in score], inf="".join(str(int(v)) for v in b["informal"]),
        order=order.tolist(), city=[simplified(r, CITY_TOL) for r in b["rings"]],
        box=[round(cx - half), round(cy - half), round(cx + half), round(cy + half)],
        near=near, local=[simplified(b["rings"][i], LOCAL_TOL) for i in near], fires=fires)
    if sum(v >= floor["value"] for v in data["s"]) != floor["n"]:
        raise AssertionError("the rounded scores' pool at the floor is not the bundle's")
    out.write_text(json.dumps(data, separators=(",", ":")))
    rank = int((score > score[follow]).sum()) + 1
    return dict(score=float(score[follow]), rank=rank, n_blocks=len(score),
                pool=int((score >= floor["value"]).sum()), floor=floor["value"],
                precision=floor["precision"], recall=floor["recall"])


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


def page() -> None:
    key, src = METHOD.pick.key(), METHOD.pick.source()
    rows = src.rows(key)
    secs = METHOD.runtime.seconds(key, src, rows)
    fs = ["000"] + list(dict.fromkeys(r["f"] for r in rows if r is not None))
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
    lat, lon = centroid_wgs84()
    sc = screen_bake(OUT / "screen.json")
    if sc["pool"] != next(f["n"] for f in json.loads(SCREEN.read_text())["floors"]
                          if f["metric"] == SCREEN_METRIC):
        raise AssertionError("the screen's pool at its floor is not the bundle's")
    n_homes = len(np.load(cp.DATA / "baseline_times.npz")["u"])
    ex = extrapolate(TOPOLOGY_RUNS, n_homes)
    total = max(s for s in secs if s is not None)
    w = world()
    data = dict(frames=frames, sheet=dict(n=len(fs), w=size[0], h=size[1]),
                stem=METHOD.stem, default=DEFAULT, iso=ISO_STEP)
    maps = (f"https://www.google.com/maps/@?api=1&map_action=map&center={lat:.5f}%2C{lon:.5f}"
            f"&zoom=17&basemap=satellite")
    runs = "; ".join(f"{r.n:,} buildings {r.seconds:,.1f} s" for r in TOPOLOGY_RUNS)
    fields = {
        "NAME": htmllib.escape(METHOD.name),
        "BLURB": htmllib.escape(METHOD.blurb),
        "BID": cp.BID,
        "LAT": f"{abs(lat):.4f}° {'S' if lat < 0 else 'N'}",
        "LON": f"{abs(lon):.4f}° {'E' if lon >= 0 else 'W'}",
        "MAPS": htmllib.escape(maps),
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
        "SCALE": f"{w.scale:.0f}",
        "STEM": METHOD.stem,
        "DATA": json.dumps(data, separators=(",", ":")),
        "SC_SCORE": f"{sc['score']:.4f}",
        "SC_FLOOR": f"{sc['floor']:g}",
        "SC_TIMES": f"{sc['score'] / sc['floor']:.1f}",
        "SC_RANK": f"{sc['rank']:,}",
        "SC_N": f"{sc['n_blocks']:,}",
        "SC_TOP": f"{100 * sc['rank'] / sc['n_blocks']:.2g}",
        "SC_POOL": f"{sc['pool']:,}",
        "SC_PREC": f"{100 * sc['precision']:.0f}",
        "SC_REC": f"{100 * sc['recall']:.0f}",
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


def main(argv: list[str]) -> None:
    matplotlib.use("Agg")
    match argv:
        case ["plates"]:
            plates()
        case ["page"]:
            page()
        case []:
            plates()
            page()
        case _:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
