"""Block 5810's road networks built out by displacement, under RUNTIME, painted for the page's
slider: per frame the current |j| and the power dissipated per m^2 (flow5810/compute.py's energy,
render.py's paint at 1200 px over the same extent), the lenses, each home's escape time, the sheet
conductance and the energy identity.

    S=research/roadless/atlas/roads5810/compute.py     # from the repo root
    PYTHONPATH=. uv run python $S baseline            # the plain mesh's baseline (the shared D 0)
    PYTHONPATH=. uv run python $S <method> <pass>     # cycle_native|clearance_looped coarse|fine

A network (Network, a Strategy: Roads here, the lineup's and the studies' road networks; a grown
region's steps and an optimizer's answer per budget in networks.py) is its prefixes in build
order, each a paved ground and the buildings cleared beside it. Its frames are the canonical
prefixes (Network.prefix_len under common.MEASURE, area weights: the SHORTEST prefix whose D
reaches the target) at its pass's targets (PASSES) where the whole network reaches them, and
(`full`) the whole network. Its corridors are carved into one fixed mesh as
road_rescore.network_rows carves the lineup's (built in `meshes`, kept for a rerun), each prefix
a road state (Model.road_state, or Model.cleared_road_state beside a clearing: the corridor's
atoms open, its triangles at the proposed road's boost), and the mesh's own baseline (no road)
the gains' denominator. Writes into `data` (DATA for the baseline and the lineup's two networks,
a study's own folder for its networks):
<method>_<run>.npz (the rasters, the atoms and per atom the stop from which it is open, carved or
its building cleared, the corridors, each building's ramped share), <method>_<run>.json,
<method>_<run>_times.npz (each home's escape time, the Time plates') and <method>_<run>_sheet.json
(the sheet conductance per stop, sheet.py's `C`). The rasters stay in the site's scratch (DATA);
only the plates render.py draws from them live beside the page.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, Protocol

import numpy as np
import shapely
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[4]
SITE = ROOT / ".superpowers" / "sdd" / "2026-10-22-site"     # the site's scratch
FLOW = SITE / "flow5810"      # the Block 5810 Pipeline page's flow plates: their energy and paint
DATA = SITE / "roads5810"     # the baseline's and the lineup's two networks' rasters, and meshes
sys.path.insert(0, str(ROOT / "research" / "roadless"))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402
import gap_maps as gm  # noqa: E402
import geopandas as gpd  # noqa: E402
import metrics  # noqa: E402
import road_rescore as rr  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402

BID = "ZAF.9.3.1_1_5810"
LINEUP = ROOT / "research/roadless/road_rows/lineup-f8fa4e53"
class Pass(NamedTuple):
    """A run's frames: its targets, whether the whole network is one."""
    targets: tuple[float, ...]
    full: bool


# coarse: the first slider (5% steps to the whole network, its 0.20 and up kept as extras); fine:
# the owner's slider of 2026-10-22, D 0.01 to 0.15 by 0.01, less coarse's 0.05, 0.10 and 0.15
PASSES = {"coarse": Pass(targets=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35), full=True),
          "fine": Pass(targets=tuple(round(0.01 * i, 2) for i in range(1, 16) if i % 5),
                       full=False)}
WIDE = 1200            # the plates' width in pixels (flow5810's 1600; the same extent)
TOP = 0.05             # the open ground's densest share whose power is reported


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


flow = load("flow5810_compute", FLOW / "compute.py")
rnd = load("flow5810_render", FLOW / "render.py")
rnd.WIDE = WIDE        # full_frame reads it: the pad scales with the block, the extent the same


def frame_of(m: gcl.Model):  # type: ignore[no-untyped-def]
    return rnd.full_frame(m.geo.boundary)


def paint_state(m: gcl.Model, st: gcl.State, fr) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """The state's flow under the population's demand, its power per triangle (flow5810's
    energy), the rasters of |j| and the power density, the energy identity and the top share."""
    fl = gm.flow_of(m, st, m.pop)
    e = flow.energy(m, st, fl)
    c = st.sys.cells
    xy, tri = st.fm.xy, st.fm.tri
    live = st.reach.live
    ta = 0.5 * np.abs(gm._cross(xy, tri))
    jmag = np.hypot(fl.j[:, 0], fl.j[:, 1])
    lt = live[c.cell]
    jt = np.where(lt, jmag[c.cell], np.nan)
    pt = np.where(lt, e["trip"] / np.maximum(ta, 1e-300), np.nan)
    return dict(
        j=rnd.paint(xy, tri, jt, fr).astype(np.float32),
        p=rnd.paint(xy, tri, pt, fr).astype(np.float32),
        jt=jt, pt=pt, ta=ta, lt=lt,
        P_energy=e["P"], power_cells=e["total"], power_tri=e["tri_total"],
        rel_err=(e["total"] - e["P"]) / e["P"], rel_err_tri=(e["tri_total"] - e["P"]) / e["P"],
        kirchhoff_max=float(np.abs(gm.conservation(st, fl)).max()),
        top5_power=rnd.top_share(pt[lt], ta[lt], TOP),
        top5_current=rnd.top_share(jt[lt], ta[lt], TOP),
        open_area=float(ta[lt].sum()), unknowns=int(st.reach.N))


SCALARS = ("P_energy", "power_cells", "power_tri", "rel_err", "rel_err_tri", "kirchhoff_max",
           "top5_power", "top5_current", "open_area", "unknowns")


def baseline(*, data: Path) -> None:
    """The plain mesh's baseline (the clearing plates' `before`): its rasters, the area-weighted
    medians of |j| and the power density over its open ground (the plates' scales), its lenses."""
    t0 = time.time()
    metric = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU())
    [b] = common.build_blocks([BID])
    m = metric.clearing(b, demand=common.PopulationDemand())
    t_model = time.time() - t0
    fr = frame_of(m)
    st = m.state(np.zeros(m.n))
    s = paint_state(m, st, fr)
    lt, ta = s["lt"], s["ta"]
    mj = rnd.wmedian(s["jt"][lt], ta[lt])
    mp = rnd.wmedian(s["pt"][lt], ta[lt])
    polys = m.geo.polys
    np.savez(data / "baseline.npz", j=s["j"], p=s["p"], polys=shapely.to_wkb(polys),
             boundary=shapely.to_wkb(np.asarray([m.geo.boundary], dtype=object)),
             streets=shapely.to_wkb(np.asarray([m.geo.streets], dtype=object)),
             frame=np.asarray(fr.frame, dtype=float), lim=np.asarray(fr.lim, dtype=float))
    info = dict(metric=metric.name, block=BID, frame=fr.frame._asdict(), median_j=mj, median_p=mp,
                P=m.base.P, Pq=m.base.Pq, J2=m.base.J2, n_enclosed=m.base.n_enclosed,
                **{k: s[k] for k in SCALARS}, t_model=t_model, t_total=time.time() - t0)
    (data / "baseline.json").write_text(json.dumps(info, indent=1))
    np.savez(data / "baseline_times.npz", u=m.base.u, live=m.base.live)     # the Time plates'
    print(json.dumps(info, indent=1), flush=True)


class Network(Protocol):
    """A page network: its prefixes 1 .. `size` in build order, each a paved ground (world frame,
    carved as a proposed road) and the buildings cleared beside it (nested in the prefix)."""

    @property
    def size(self) -> int:
        """The whole network's prefix."""
        ...

    def prefix_len(self, target: float) -> int:
        """The shortest prefix whose D (area weights, common.MEASURE) reaches `target`, or
        `size`."""
        ...

    def corridor(self, mm: int) -> Any:
        """Prefix mm's paved ground, world frame; nested in mm."""
        ...

    def cleared(self, mm: int) -> NDArray[np.bool_]:
        """Per building whether it is cleared beside prefix mm; nested in mm."""
        ...

    def c(self, mm: int) -> NDArray[np.float64]:
        """Per building its ramped destroyed share under prefix mm and its clearing
        (common.MEASURE)."""
        ...

    def roads(self, mm: int) -> tuple[int, float]:
        """Prefix mm's roads: (pieces, metres)."""
        ...


class Roads:
    """A road network already in its build order (the lineup's canonical order, or a re-solved
    build's own: each stage's roads after the last's): prefix mm its first mm roads, their
    corridor (road_rescore.corridor_of), no clearing."""

    def __init__(self, b, ordered: gpd.GeoDataFrame) -> None:  # type: ignore[no-untyped-def]
        self.b, self.ordered = b, ordered
        self.W = common.POPULATIONS["area"].weights(np.asarray(b.buildings.outlines))

    @property
    def size(self) -> int:
        return len(self.ordered)

    def prefix_len(self, target: float) -> int:
        return rr.prefix_len(self.b, self.ordered, target, self.W, measure=common.MEASURE)

    def corridor(self, mm: int) -> Any:
        return rr.corridor_of(self.ordered, mm)

    def cleared(self, mm: int) -> NDArray[np.bool_]:
        return np.zeros(len(self.W), dtype=bool)

    def c(self, mm: int) -> NDArray[np.float64]:
        """common.MEASURE's shares under the prefix's corridor, its D checked against
        common.displacement's."""
        pre = self.ordered.iloc[:mm]
        cdes = np.asarray(common.MEASURE.c(self.b.buildings, road_corridor(pre)), dtype=float)
        D = float((cdes * self.W).sum() / self.W.sum())
        D_ref = common.displacement(self.b, pre, self.W, measure=common.MEASURE)
        if abs(D - D_ref) > 1e-12:
            raise AssertionError(f"prefix {mm}: D {D} against common.displacement's {D_ref}")
        return cdes

    def roads(self, mm: int) -> tuple[int, float]:
        return mm, float(self.ordered.iloc[:mm].geometry.length.sum())


def network(name: str, run: str, *, frames: Pass, networks: Path, data: Path,
            meshes: Path) -> None:
    """The frames of the network `name` stored in `networks` (<name>.parquet and its canonical
    order, <name>.ordered.parquet), named <name>_<run> in `data`."""
    t0 = time.time()
    [b] = common.build_blocks([BID])
    net = gpd.read_parquet(networks / f"{name}.parquet")
    ordered = gpd.read_parquet(networks / f"{name}.ordered.parquet")
    again = common.canonical_order(b, net)
    if len(again) != len(ordered) or not again.geometry.geom_equals_exact(
            ordered.geometry, tolerance=0.0).all():
        raise AssertionError(f"{name}: the stored order is not common.canonical_order's")
    frames_of(name, run, b, Roads(b, ordered), t0, frames=frames, data=data, meshes=meshes)


def opened(carved: NDArray[np.int64], parent: NDArray[np.int64],
           cleared: list[NDArray[np.bool_]]) -> NDArray[np.int64]:
    """Per atom the first stop (1 ..) from which it is open: carved by the stop's corridor
    (`carved`, the carving's stop, 0 never) or its building (`parent`) cleared beside it; 0
    never. Refuses clearings that are not nested."""
    for k in range(1, len(cleared)):
        if (cleared[k - 1] & ~cleared[k]).any():
            raise ValueError(f"stop {k + 1}'s clearing drops a building stop {k} cleared")
    out = carved.copy()
    for k in range(len(cleared), 0, -1):
        hit = cleared[k - 1][parent]
        out[hit & ((out == 0) | (out > k))] = k
    return out


def frames_of(name: str, run: str, b, net: Network, t0: float, *,  # type: ignore[no-untyped-def]
              frames: Pass, data: Path, meshes: Path) -> None:
    """`network`'s frames of the Network `net` at `frames`' targets: per stop its rasters, lenses,
    escape times and sheet (module docstring)."""
    metric = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU())
    W = common.POPULATIONS["area"].weights(np.asarray(b.buildings.outlines))
    D_full = float((net.c(net.size) * W).sum() / W.sum())
    want = [(t, net.prefix_len(t)) for t in frames.targets if D_full >= t]
    if frames.full:
        want.append((None, net.size))
    stops = sorted({mm for _t, mm in want})
    t_cut = time.time() - t0
    print(f"{name}: D_full {D_full:.4f}, stops {stops} ({t_cut:.0f}s)", flush=True)

    geo = gc.geometry(b)
    carving = gcl.Carving.of(geo, [[geo.local(net.corridor(mm)) for mm in stops]])
    meshes.mkdir(exist_ok=True)
    fx = metric.fixed(b, cache=meshes, carving=carving)
    t_mesh = time.time() - t0 - t_cut
    m = metric.model(b, fx, demand=common.PopulationDemand())
    t_model = time.time() - t0 - t_cut - t_mesh
    if len(m.geo.polys) != len(b.buildings.outlines):
        raise AssertionError("the model's buildings are not the block's")
    print(f"{name}: mesh {fx.fm.nt} triangles, {carving.n} atoms, mesh {t_mesh:.0f}s, "
          f"model {t_model:.0f}s, P0 {m.base.P:.6e}", flush=True)
    fr = frame_of(m)
    cleared = [net.cleared(mm) for mm in stops]
    out: dict[str, np.ndarray] = dict(
        atoms=shapely.to_wkb(np.asarray(m.carving.polys, dtype=object)),
        atom_stop=opened(m.carving.stop[:, 0], m.carving.parent, cleared),
        stops=np.asarray(stops),
        frame=np.asarray(fr.frame, dtype=float), lim=np.asarray(fr.lim, dtype=float))
    times: dict[str, np.ndarray] = dict(stops=np.asarray(stops), u_base=m.base.u,
                                        live_base=m.base.live)
    ts = time.time()
    st0 = m.state(np.zeros(m.n))
    sheets: dict[str, Any] = dict(base=m.sheet(st0), stops={})
    s0 = paint_state(m, st0, fr)
    del st0
    out["j_base"], out["p_base"] = s0["j"], s0["p"]
    base = dict(P=m.base.P, Pq=m.base.Pq, J2=m.base.J2, **{k: s0[k] for k in SCALARS},
                t=time.time() - ts)
    print(f"{name} base: {json.dumps(base)}", flush=True)
    del s0
    rows = []
    for k, (mm, cl) in enumerate(zip(stops, cleared, strict=True), start=1):
        ts = time.time()
        cdes = net.c(mm)
        D = float((cdes * W).sum() / W.sum())
        prefix = gcl.Prefix(net=0, stop=k)
        st = m.cleared_road_state(m.k_of(cl), prefix) if cl.any() else m.road_state(prefix)
        L = m.lenses_of(st)
        sheets["stops"][str(mm)] = m.sheet(st)
        s = paint_state(m, st, fr)
        del st
        n_roads, road_m = net.roads(mm)
        targets = [t for t, x in want if x == mm]
        row = dict(m=mm, targets=targets, n_roads=n_roads, road_m=road_m, n_cleared=int(cl.sum()),
                   D_area=D, P=L.P, Pq=L.Pq, J2=L.J2,
                   **{kk: v for kk, v in rr._lens_row(m, L).items()},
                   **{kk: s[kk] for kk in SCALARS if kk != "unknowns"}, t=time.time() - ts)
        rows.append(row)
        out[f"j_{mm}"], out[f"p_{mm}"] = s["j"], s["p"]
        out[f"c_{mm}"] = cdes
        out[f"corridor_{mm}"] = shapely.to_wkb(np.asarray([m.carving.corridors[0][k - 1]],
                                                          dtype=object))
        times[f"u_{mm}"], times[f"live_{mm}"] = L.u, L.live
        print(f"{name} m {mm}: {json.dumps(row)}", flush=True)
        del s
    np.savez(data / f"{name}_{run}.npz", **out)
    np.savez(data / f"{name}_{run}_times.npz", **times)
    (data / f"{name}_{run}_sheet.json").write_text(json.dumps(sheets, indent=1))
    info = dict(method=name, metric=metric.name, block=BID, D_full=D_full,
                rule="the shortest prefix whose D_area reaches the target (Network.prefix_len: "
                     "the lineup's road_rescore.prefix_len for roads)",
                want=[dict(target=t, m=mm) for t, mm in want], nt=int(fx.fm.nt),
                atoms=int(carving.n), base=base, frames=rows, t_cut=t_cut, t_mesh=t_mesh,
                t_model=t_model, t_total=time.time() - t0,
                peak_rss_gb=rr._status_gb("VmHWM"))
    (data / f"{name}_{run}.json").write_text(json.dumps(info, indent=1))
    print(f"{name}: done in {time.time() - t0:.0f}s, peak {info['peak_rss_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    arg, *rest = sys.argv[1:]
    if arg == "baseline":
        baseline(data=DATA)
    elif arg in ("cycle_native", "clearance_looped") and len(rest) == 1 and rest[0] in PASSES:
        network(arg, rest[0], frames=PASSES[rest[0]], networks=LINEUP / "scratch/networks" / BID,
                data=DATA, meshes=DATA / ".meshes")
    else:
        raise SystemExit(f"{sys.argv[1:]}: baseline, or cycle_native|clearance_looped coarse|fine")
