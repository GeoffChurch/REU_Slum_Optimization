"""Plates from compute.py's rasters, drawn as flow5810/render.py draws the clearing's (its plate,
colour maps, scales and colour bars, reused): per frame the network, the current |j|, the power
per m^2, their gains over the baseline and each home's escape time over the baseline's (Time);
then frames.json.

    S=research/roadless/atlas/roads5810/render.py      # from the repo root
    PYTHONPATH=. uv run python $S plates baseline
    PYTHONPATH=. uv run python $S plates <method> <pass> <data dir>  # compute.DATA, or a study's
    PYTHONPATH=. uv run python $S index                   # frames.json, the colour bars
    PYTHONPATH=. uv run python $S times <method> <pass> <data dir>  # the Time plates

Scales: |j| and the power density on flow5810's log scales relative to the PLAIN mesh's baseline
(the shared D 0: the area-weighted medians of its open ground). Gains (owner 2026-10-22: "can it
show decreases in e.g. red?"): log2 of a frame over its own mesh's baseline (the network's carved
mesh with no road, so no mesh difference reads as a gain), from -1 (1/2x, red: the flow or power
drawn away) through 1x (a pale neutral) to +1 (2x and more, blue; newly open ground at the top),
no colour off the open ground (owner 2026-10-26: the range narrowed from 1/8x -- 8x, so the
interior's changes register). Time (owner 2026-10-26): each home shaded by its escape time over
its baseline's on the gains' scale (red: faster), since the current and the power barely change
where a clearing or road far off lowers a home's potential; the baseline's own plate its escape
time, linear to the 99th percentile; a home with none (cleared, destroyed, or no demand) open.
The network: every building grey, shaded to the clearing plates' orange by its ramped destroyed
share c_i (0 to 1; a building cleared beside the roads 1), the roads' corridor over them.
The block's edge on every plate as gap_maps.draw_edge draws it (owner 2026-10-28: the north edge
is a railway): the street dark, a railway along it a line with cross-ticks, broken at its
crossings, water blue; from the page metric's own barriers (`edge`).
Each plate is written as lossy WebP into `plates` (PLATES, beside the page: every plate it
names, local.py's check); a network's rasters are read from `data` (compute.DATA, or a study's
own folder), the plain baseline always from compute.DATA.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute as cp  # noqa: E402  (flow5810's modules, loaded once; rnd.WIDE set to 1200)
import geopandas as gpd  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from PIL import Image  # noqa: E402

rnd, gm = cp.rnd, cp.gm
PLATES = HERE / "plates"         # beside the page, git-ignored (local.py keeps the same path)
METHODS = ("cycle_native", "clearance_looped")
PASSES = ("coarse", "fine")      # compute.PASSES
SLIDER = tuple(round(0.01 * i, 2) for i in range(1, 16))     # the owner's slider, D 0.01 .. 0.15
EXT = "webp"
ROAD = "#475569"          # a corridor on the network plate (the streets are rnd.STREET)
STORED = (cp.LINEUP / ("cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a+xB8c3Lo6214b562S10hliving_street-"
                       "motorway-primary-residential-secondary-service-tertiary-trunk-"
                       "unclassifiedva03a092b") / "cpu/area_D0.02-0.05-0.1-0.15-0.2")
DESTROYED = mcolors.LinearSegmentedColormap.from_list("destroyed", [rnd.BUILDING, rnd.CLEARED])
LIM = 1.0                 # the gains' log2 range: 1/2x to 2x
DIV = mcolors.LinearSegmentedColormap.from_list(
    "gain_div", ["#b2182b", "#ef8a62", "#fddbc7", "#ebe9e4", "#d1e5f0", "#67a9cf", "#2166ac"])
GAIN = mcolors.Normalize(-LIM, LIM, clip=True)


WEBP_Q = 85         # lossy WebP's quality (EXT webp)


def small(png: Path, out: Path) -> None:
    """`png` written small at `out`: a 256-colour palette PNG (median cut, no dither) for .png,
    lossy WebP at WEBP_Q for .webp."""
    im = Image.open(png).convert("RGB")
    if out.suffix == ".png":
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.NONE).save(out, optimize=True)
    elif out.suffix == ".webp":
        im.save(out, "WEBP", quality=WEBP_Q, method=6)
    else:
        raise ValueError(f"{out}: .png or .webp")
    png.unlink()
    print(out, f"{out.stat().st_size / 1024:.0f} KB", flush=True)


def scratch(out: Path) -> Path:
    """The full-size PNG `small` reads, beside `out`."""
    return out.with_name(f".{out.stem}.png")


def edge(stored) -> gm.Edge:  # type: ignore[no-untyped-def]
    """The block's edge as the plates draw it (gm.edge_of) under the page metric's own
    barriers (metrics.RUNTIME's); raises unless its polygon is the `stored` baseline's (the
    plates' frame: baseline.npz)."""
    [b] = cp.common.build_blocks([cp.BID])
    e = gm.edge_of(b, barriers=cp.gcl.complex_of(
        cp.metrics.RUNTIME, factorization=cp.gap_factor.SuperLU()).exits.barriers)
    [boundary] = shapely.from_wkb(stored["boundary"])
    if not shapely.equals(e.boundary, boundary):
        raise AssertionError("the metric's cut block is not the stored baseline's")
    return e


def draw_edge(ax, fr, e: gm.Edge) -> None:  # type: ignore[no-untyped-def]
    """`e` on a plate of frame `fr`, in flow5810's colours (rnd.STREET, rnd.EDGE)."""
    gm.draw_edge(ax, e, px=fr.frame.h, ink=rnd.STREET, faint=rnd.EDGE, street_lw=2.0)


def frame_from(d) -> rnd.Frame:  # type: ignore[no-untyped-def]
    x0, y0, h, ny, nx = d["frame"]
    f = gm.Frame(x0=float(x0), y0=float(y0), h=float(h), ny=int(ny), nx=int(nx))
    return rnd.Frame(f, tuple(float(v) for v in d["lim"]))


def norms(base: dict) -> tuple:  # type: ignore[type-arg]
    mj, mp = base["median_j"], base["median_p"]
    return (mcolors.LogNorm(rnd.J_LO * mj, rnd.J_HI * mj, clip=True),
            mcolors.LogNorm(rnd.P_LO * mp, rnd.P_HI * mp, clip=True))


@contextmanager
def ramp_of(cmap) -> Iterator[None]:  # type: ignore[no-untyped-def]
    """flow5810's colour bar draws with its module's RAMP: `cmap` in its place meanwhile."""
    saved = rnd.RAMP
    rnd.RAMP = cmap
    try:
        yield
    finally:
        rnd.RAMP = saved


def flow_plate(img, norm, fr, polys, k, *, ramp, outline, edge: gm.Edge, out: Path) -> None:  # type: ignore[no-untyped-def]
    """rnd.plate's figure of `img` on `ramp` (rnd.RAMP for a flow, DIV for a gain) at `out`: the
    standing buildings grey, the cleared (k 1) outlined if `outline`, the edge (draw_edge)."""
    f = fr.frame
    fig = plt.figure(figsize=(f.nx / 100, f.ny / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, origin="lower", extent=f.extent, interpolation="nearest", cmap=ramp,
              norm=norm)
    gpd.GeoSeries(list(polys[k < 1.0])).plot(ax=ax, color=rnd.BUILDING, lw=0)
    if outline:
        gpd.GeoSeries(list(polys[k >= 1.0])).boundary.plot(ax=ax, color=rnd.CLEARED, lw=0.7)
    draw_edge(ax, fr, edge)
    ax.set_xlim(fr.lim[0], fr.lim[1])
    ax.set_ylim(fr.lim[2], fr.lim[3])
    ax.set_axis_off()
    tmp = scratch(out)
    fig.savefig(tmp, dpi=100, facecolor="white")
    plt.close(fig)
    small(tmp, out)


def network_plate(fr, polys, c, corridor, *, edge: gm.Edge, out: Path) -> None:  # type: ignore[no-untyped-def]
    """rnd.plate's figure with no flow: the buildings shaded by c, the corridor, the edge
    (draw_edge)."""
    f = fr.frame
    fig = plt.figure(figsize=(f.nx / 100, f.ny / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    gpd.GeoSeries(list(polys)).plot(ax=ax, color=DESTROYED(np.clip(c, 0.0, 1.0)), lw=0)
    if corridor is not None:
        gpd.GeoSeries([corridor]).plot(ax=ax, color=ROAD, lw=0)
    draw_edge(ax, fr, edge)
    ax.set_xlim(fr.lim[0], fr.lim[1])
    ax.set_ylim(fr.lim[2], fr.lim[3])
    ax.set_axis_off()
    tmp = scratch(out)
    fig.savefig(tmp, dpi=100, facecolor="white")
    plt.close(fig)
    small(tmp, out)


def time_plate(fr, polys, value, *, norm, ramp, gone, corridor, edge: gm.Edge,  # type: ignore[no-untyped-def]
               out: Path) -> None:
    """Each home filled by `value` on `ramp` (nan: no home, drawn open with an outline), the
    `gone` buildings (cleared or destroyed) in ROAD's grey, the corridor, the edge (draw_edge)."""
    f = fr.frame
    fig = plt.figure(figsize=(f.nx / 100, f.ny / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    polys = np.asarray(polys, dtype=object)
    has = np.isfinite(value) & ~gone
    gpd.GeoSeries(list(polys[has])).plot(ax=ax, color=ramp(norm(value[has])), lw=0)
    gpd.GeoSeries(list(polys[~has & ~gone])).boundary.plot(ax=ax, color=rnd.BUILDING, lw=0.4)
    gpd.GeoSeries(list(polys[gone])).plot(ax=ax, color=ROAD, lw=0)
    if corridor is not None:
        gpd.GeoSeries([corridor]).plot(ax=ax, color=ROAD, lw=0)
    draw_edge(ax, fr, edge)
    ax.set_xlim(fr.lim[0], fr.lim[1])
    ax.set_ylim(fr.lim[2], fr.lim[3])
    ax.set_axis_off()
    tmp = scratch(out)
    fig.savefig(tmp, dpi=100, facecolor="white")
    plt.close(fig)
    small(tmp, out)


def time_ratio(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    """log2 of each home's escape time over its baseline's, clipped to the gains' range; a home
    freed (no finite time before, one after) at -LIM; nan where it has no time after."""
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(np.isfinite(before) & (before > 0), np.log2(after / before), -LIM)
    return np.where(np.isfinite(after), np.clip(g, -LIM, LIM), np.nan)


def time_norm(u: np.ndarray) -> mcolors.Normalize:
    """The baseline Time plate's scale: linear from 0 to the homes' 99th percentile."""
    return mcolors.Normalize(0.0, float(np.percentile(u[np.isfinite(u)], 99)), clip=True)


def plates_times(method: str, run: str, *, data: Path, plates: Path) -> None:
    """The Time plate of each of <method>_<run>'s frames (compute.frames_of's escape times over
    its own mesh's baseline), the corridor over the homes, into `plates`."""
    info = json.loads((data / f"{method}_{run}.json").read_text())
    d = np.load(data / f"{method}_{run}.npz", allow_pickle=True)          # our own file: WKB
    t = np.load(data / f"{method}_{run}_times.npz")
    b = np.load(cp.DATA / "baseline.npz", allow_pickle=True)
    if not np.array_equal(t["stops"], d["stops"]):
        raise AssertionError(f"{method}_{run}: the escape times' stops are not the frames'")
    fr = frame_from(d)
    polys = shapely.from_wkb(b["polys"])
    e = edge(b)
    names = frame_names(info)
    for mm in [int(s) for s in d["stops"]]:
        [corridor] = shapely.from_wkb(d[f"corridor_{mm}"])
        time_plate(fr, polys, time_ratio(t[f"u_{mm}"], t["u_base"]), norm=GAIN, ramp=DIV,
                   gone=d[f"c_{mm}"] >= 1.0, corridor=corridor, edge=e,
                   out=plates / f"{method}_{names[mm]}_time.{EXT}")


def divgain(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log2(a / b) where both are open; newly open ground (b not open, a open) at +LIM; NaN where
    a is not open ground."""
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(np.isfinite(b) & (b > 0), np.log2(a / b), LIM)
    g = np.where(np.isfinite(a) & (a > 0), g, np.nan)
    return np.clip(g, -LIM, LIM)


def tag(D: float) -> str:
    return f"{int(round(D * 1000)):03d}"


def frame_names(info: dict) -> dict[int, str]:  # type: ignore[type-arg]
    """Per stop m its frame's DDD: its target's, or the whole network's own D."""
    out = {}
    for w in info["want"]:
        D = w["target"] if w["target"] is not None else info["D_full"]
        out.setdefault(w["m"], tag(D))
    return out


def plates_baseline(*, plates: Path) -> None:
    base = json.loads((cp.DATA / "baseline.json").read_text())
    d = np.load(cp.DATA / "baseline.npz", allow_pickle=True)          # our own file: WKB as objects
    fr = frame_from(d)
    polys = shapely.from_wkb(d["polys"])
    e = edge(d)
    jn, pn = norms(base)
    k = np.zeros(len(polys))
    network_plate(fr, polys, k, None, edge=e, out=plates / f"baseline_network.{EXT}")
    for name, key, norm in (("current", "j", jn), ("power", "p", pn)):
        flow_plate(d[key], norm, fr, polys, k, ramp=rnd.RAMP, outline=False, edge=e,
                   out=plates / f"baseline_{name}.{EXT}")
    u = np.load(cp.DATA / "baseline_times.npz")["u"]
    time_plate(fr, polys, u, norm=time_norm(u), ramp=rnd.RAMP, gone=np.zeros(len(polys), bool),
               corridor=None, edge=e, out=plates / f"baseline_time.{EXT}")


def plates_method(method: str, run: str, *, data: Path, plates: Path) -> None:
    """The plates of <method>_<run>'s frames (compute.frames_of's files in `data`) into `plates`."""
    base = json.loads((cp.DATA / "baseline.json").read_text())
    info = json.loads((data / f"{method}_{run}.json").read_text())
    d = np.load(data / f"{method}_{run}.npz", allow_pickle=True)    # our own file: WKB as objects
    b = np.load(cp.DATA / "baseline.npz", allow_pickle=True)
    fr = frame_from(d)
    if not np.allclose(d["frame"], b["frame"]) or not np.allclose(d["lim"], b["lim"]):
        raise AssertionError("the method's frame is not the baseline's")
    polys = shapely.from_wkb(b["polys"])
    e = edge(b)
    atoms = shapely.from_wkb(d["atoms"])
    astop = d["atom_stop"]          # per atom the stop from which it is open: carved or cleared
    jn, pn = norms(base)
    names = frame_names(info)
    for k, mm in enumerate([int(s) for s in d["stops"]], start=1):
        DDD = names[mm]
        gone = ((astop >= 1) & (astop <= k)).astype(float)
        [corridor] = shapely.from_wkb(d[f"corridor_{mm}"])
        stem = f"{method}_{DDD}"
        network_plate(fr, polys, d[f"c_{mm}"], corridor, edge=e,
                      out=plates / f"{stem}_network.{EXT}")
        for name, key, norm in (("current", "j", jn), ("power", "p", pn)):
            img = d[f"{key}_{mm}"]
            flow_plate(img, norm, fr, atoms, gone, ramp=rnd.RAMP, outline=True, edge=e,
                       out=plates / f"{stem}_{name}.{EXT}")
            flow_plate(divgain(img, d[f"{key}_base"]), GAIN, fr, atoms, gone, ramp=DIV,
                       outline=True, edge=e, out=plates / f"{stem}_{name}_gain.{EXT}")


def colorbar_with(cmap, norm, ticks, labels, text: str, path: Path) -> None:  # type: ignore[no-untyped-def]
    """rnd.colorbar with another colour map."""
    with ramp_of(cmap):
        rnd.colorbar(norm, ticks, labels, text, path)


def index(*, data: Path, plates: Path) -> None:
    base = json.loads((data / "baseline.json").read_text())
    flow_stats = json.loads((cp.FLOW / "stats.json").read_text())
    jn, pn = norms(base)
    same = (np.isclose(base["median_j"], flow_stats["median_j_before"], rtol=1e-9)
            and np.isclose(base["median_p"], flow_stats["median_p_before"], rtol=1e-9))
    bars = {}
    if same:          # the same norms: flow5810's bars are these
        for n in ("current", "power"):
            shutil.copy(cp.FLOW / f"{n}_colorbar.png", plates / f"{n}_colorbar.png")
            bars[n] = f"copied from flow5810/{n}_colorbar.png (identical scale)"
    else:
        raise SystemExit(f"the baseline's medians {base['median_j']}, {base['median_p']} are not "
                         f"flow5810's: draw new bars")
    colorbar_with(DIV, GAIN, [-1, -0.5, 0, 0.5, 1], ["1/2x", "0.71x", "1x", "1.41x", "2x+ or new"],
                  "after over before (log): red drawn away or faster, blue raised or slower",
                  plates / "gain_colorbar.png")
    u = np.load(data / "baseline_times.npz")["u"]
    un = time_norm(u)
    colorbar_with(rnd.RAMP, un, [0, un.vmax / 2, un.vmax], ["0", f"{un.vmax / 2:.0f}",
                  f"{un.vmax:.0f}+"], "a home's escape time with nothing built or cleared",
                  plates / "time_colorbar.png")
    bars["gain"] = "new: log2 of the frame over its baseline, diverging, 1/2x to 2x"
    bars["time"] = "new: the baseline's escape time per home, linear to its 99th percentile"
    colorbar_with(DESTROYED, mcolors.Normalize(0.0, 1.0), [0, 0.25, 0.5, 0.75, 1.0],
                  ["0", "0.25", "0.5", "0.75", "1"],
                  "a building's ramped destroyed share c_i (destroyed area / 0.10, at most 1)",
                  plates / "network_colorbar.png")
    bars["network"] = "new"

    def row(r: dict, method: str, DDD: str) -> dict:  # type: ignore[type-arg]
        slider = any(t is not None and round(t, 2) in SLIDER for t in r["targets"])
        return dict(frame=DDD, slider=slider, files={k: f"{method}_{DDD}_{k}.{EXT}" for k in
                                      ("network", "current", "power", "current_gain",
                                       "power_gain")},
                    targets=r["targets"], prefix_m=r["m"], D_area=r["D_area"],
                    n_roads=r["n_roads"], road_m=r["road_m"], P=r["P"], L_P=r["L_P"],
                    L_Pq=r["L_Pq"], L_J2=r["L_J2"], L_P_vs_plain=1.0 - r["P"] / base["P"],
                    top5_power_share=r["top5_power"], top5_current_share=r["top5_current"],
                    n_enclosed=r["n_enclosed"], energy_rel_err=r["rel_err"],
                    energy_rel_err_tri=r["rel_err_tri"], kirchhoff_max=r["kirchhoff_max"],
                    unknowns=r["unknowns"], t_state=r["t"])

    methods = {}
    checks = {}
    for method in METHODS:
        infos = {run: json.loads((data / f"{method}_{run}.json").read_text()) for run in PASSES}
        info = infos["coarse"]
        frames, every = [], []
        for run, i in infos.items():
            names = frame_names(i)
            for r in i["frames"]:
                frames.append(dict(row(r, method, names[r["m"]]), run=run,
                                   P0_run_mesh=i["base"]["P"]))
                every.append(dict(r, base_P=i["base"]["P"]))
        frames.sort(key=lambda f: f["D_area"])
        stored = pd.read_parquet(STORED / f"{cp.BID}__{method}.parquet")
        cmp = []
        for _, s in stored.iterrows():
            hits = [r for r in every if r["m"] == int(s.m)]
            if not hits:
                continue
            [r] = hits
            cmp.append(dict(budget=None if s.full else float(s.budget), m=int(s.m),
                            D_area_stored=float(s.D_area), D_area=r["D_area"],
                            P_stored=float(s.P0 * (1.0 - s.L_P)), P0_stored=float(s.P0), P=r["P"],
                            P0=r["base_P"], L_P_stored=float(s.L_P), L_P=r["L_P"],
                            dL_P=r["L_P"] - float(s.L_P), L_Pq_stored=float(s.L_Pq),
                            L_Pq=r["L_Pq"], L_J2_stored=float(s.L_J2), L_J2=r["L_J2"]))
        methods[method] = dict(
            D_full=info["D_full"], n_roads_full=frames[-1]["n_roads"],
            slider=[f["frame"] for f in frames if f["slider"]],
            extras=[f["frame"] for f in frames if not f["slider"]], frames=frames,
            runs={run: dict(stops=[w["m"] for w in i["want"]], mesh_triangles=i["nt"],
                            atoms=i["atoms"], P0_carved_mesh=i["base"]["P"],
                            P0_carved_over_plain=i["base"]["P"] / base["P"],
                            base_energy_rel_err=i["base"]["rel_err"], t_mesh=i["t_mesh"],
                            t_model=i["t_model"], t_total=i["t_total"],
                            peak_rss_gb=i["peak_rss_gb"]) for run, i in infos.items()})
        checks[method] = cmp
    out = dict(
        block=cp.BID, metric=base["metric"],
        prefix_rule=("road_rescore.prefix_len under common.MEASURE (RampedDestroyed tau 0.10), "
                     "area weights: the SHORTEST canonical prefix whose D_area reaches the target "
                     "(the lineup's rule, so its rows check), the whole network where it never "
                     "does; a frame is named by its target, the whole network by its own D"),
        scales=dict(median_j=base["median_j"], median_p=base["median_p"],
                    j_range=[rnd.J_LO, rnd.J_HI], p_range=[rnd.P_LO, rnd.P_HI],
                    gain_log2_range=[-LIM, LIM], time_log2_range=[-LIM, LIM],
                    note="log, relative to the plain baseline's area-weighted median open ground "
                         "(flow5810's); gains log2 over each network's own carved-mesh baseline"),
        colorbars=bars,
        baseline=dict(frame="baseline", D_area=0.0, n_roads=0, road_m=0.0, P=base["P"],
                      L_P=0.0, L_Pq=0.0, L_J2=0.0, top5_power_share=base["top5_power"],
                      top5_current_share=base["top5_current"], energy_rel_err=base["rel_err"],
                      files={k: f"baseline_{k}.{EXT}" for k in ("network", "current", "power")}),
        methods=methods, stored_lineup_check=checks)
    (data / "frames.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(checks, indent=1), flush=True)


if __name__ == "__main__":
    PLATES.mkdir(exist_ok=True)
    if sys.argv[1:] == ["index"]:
        index(data=cp.DATA, plates=PLATES)
    elif sys.argv[1:] == ["plates", "baseline"]:
        plates_baseline(plates=PLATES)
    elif len(sys.argv) == 5 and sys.argv[1] == "plates":
        plates_method(sys.argv[2], sys.argv[3], data=Path(sys.argv[4]), plates=PLATES)
    elif len(sys.argv) == 5 and sys.argv[1] == "times":
        plates_times(sys.argv[2], sys.argv[3], data=Path(sys.argv[4]), plates=PLATES)
    else:
        raise SystemExit(__doc__)
