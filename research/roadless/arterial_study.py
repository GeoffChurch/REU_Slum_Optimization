"""The arterial study: the lineup's arterial with its displacement stop under each anchor sampler,
run through the lineup's own scoring path, and the report that decides the presets.

    S=research/roadless/arterial_study.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S run <rows> <id> <arm> [--replace]   # one (block, arm), end to end
    uv run python $S propose <rows> <id> <arm> [--replace]   # its proposal alone, unscored
    uv run python $S report <rows> <baseline-root> <ids,>

ARMS: `cur02` (PerLineFallback(32, 128), today's sampler), `cap02` (HardCap(32, 128)), both at
Destroyed(share=0.20, tau=0.10) (the lineup's ramped destroyed-area D, common.MEASURE) and
max_roads 1000, and `smoke` (HardCap, the same budget, 5 roads: the runner's own check). An arm
is the lineup's arterial (road_rescore.lineup_arms()[ARTERIAL]) with its sampler, budget and cap
replaced (method_of); its network's name carries a digest of the method's identity
(network_name), so rows of one configuration never read as another's.

`run`: the block's proposal straight from the method (never the derive cache: a cache hit would
carry no steps), timed (road_rescore.Sampler) and stepped (arterial_steps.StepLog, the steps
written after every one to <rows>/steps/<id>__<net>.parquet.partial, a name cluster pulls do not
match, and to <rows>/steps/<id>__<net>.parquet when the run ends); the network to
<rows>/scratch/networks, a proposal row (wall, CPU, peak RSS, commits, stop, roads, host) to
<rows>/proposals; then the lineup's canonical prefixes (road_rescore.prefixes_one) and their
scores on the complex under RUNTIME on a core (road_rescore.cx_network). A proposal row that
stands refuses the run unless --replace (clearings.Refuse, .Replace), or under --keep-standing
is that proposal done, the run going on to its scoring (clearings.Keep). The study's <rows> is
research/roadless/arterial_rows/study (untracked). `propose` is `run` without the scoring (a
region-scale timing, whose scoring would cost more than the proposal).

`report`: against the baseline (<baseline-root>, the lineup's rows: its proposal rows, prefixes,
complex rows and ordered networks of greedy_arterial_access_displacement), writes to
research/roadless/arterial_rows/:
- summary_runs.csv: per block and network (baseline, each arm present) the wall, commits, roads,
  stop, budget, step time (median, p90, max), anchors and candidates (max), the whole network's
  D by area under three rules (partial overlap, the lineup's ramped destroyed area, touched
  buildings counted whole: the gap between them), its scored D (count, area: ramped, from the
  prefixes), the run's own D at its stop and its difference from the same network scored afresh
  (chk_displaced: FP only), and the fidelity checks of cur02 (below);
- summary_lenses.csv: per block, network and budget (each of BUDGETS and the whole network) the
  area prefix's m, D_area, reach, L_P, L_Pq, L_J2, enclosed, and each lens minus the baseline's;
- summary_prefixes.csv: cur02 against the baseline (the same greedy, stopped earlier), per block
  and budget: both prefixes' m, the symmetric difference of their road unions (metres and share
  of the baseline prefix's length), and whether they are identical -- how far canonical_order's
  re-ordering moves the scored prefixes.
The fidelity checks of cur02: its network lies inside the baseline's (its union's length outside
the baseline's union buffered by FIDELITY_BUFFER is at most FIDELITY_TOL), and where it stopped on
no_gain it equals the baseline's (geom_equals_exact, tolerance 0). The verdict of the plan's
Ruling 7 (VERDICT_LENSES at REPORT_BUDGETS): a sampler dominates the other only if faster on every
block and its median paired difference over the blocks reaching the budget in both is >= 0 on all
six; otherwise both stay.
"""
from __future__ import annotations

import dataclasses
import hashlib
import socket
import sys
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arterial_steps  # noqa: E402
import clearings  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_factor  # noqa: E402
import road_rescore as rr  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402
from reblock.contracts import Block  # noqa: E402
from reblock.methods.arterial.anchors import AnchorSampler, HardCap, PerLineFallback  # noqa: E402
from reblock.methods.arterial.reblocker import GreedyArterialReblocker  # noqa: E402
from reblock.methods.arterial.stop import Destroyed, DisplacementBudget  # noqa: E402


class Arm(NamedTuple):
    anchors: AnchorSampler
    budget: DisplacementBudget
    max_roads: int


ARMS: dict[str, Arm] = {
    "cur02": Arm(PerLineFallback(32, 128), Destroyed(share=0.20, tau=common.MEASURE.tau), 1000),
    "cap02": Arm(HardCap(32, 128), Destroyed(share=0.20, tau=common.MEASURE.tau), 1000),
    "smoke": Arm(HardCap(32, 128), Destroyed(share=0.20, tau=common.MEASURE.tau), 5),
}
SAMPLERS = ("cur02", "cap02")               # the verdict's two arms, in Ruling 7's order (a, b)
VERDICT_LENSES = ("L_P", "L_Pq", "L_J2")
FIDELITY_BUFFER = 1e-6                       # m: the baseline union's buffer for "inside"
FIDELITY_TOL = 1e-6                          # m: road length allowed outside it
BASE_NET = rr.ARTERIAL                       # the baseline's network name
OUT = HERE / "arterial_rows"


def method_of(arm: Arm) -> GreedyArterialReblocker:
    preset = rr.lineup_arms()[rr.ARTERIAL]
    assert isinstance(preset, GreedyArterialReblocker)
    return dataclasses.replace(preset, anchors=arm.anchors, budget=arm.budget,
                               max_roads=arm.max_roads)


def network_name(arm_name: str, method: GreedyArterialReblocker) -> str:
    digest = hashlib.sha256(repr(method.identity).encode()).hexdigest()[:8]
    return f"arterial-{arm_name}-{digest}"


def _metric() -> gcl.ComplexMetric:
    import metrics
    return gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.factorization_of("cpu"))


def steps_path(rows: Path, bid: str, name: str) -> Path:
    return rows / "steps" / f"{bid}__{name}.parquet"


def run(rows: Path, bid: str, arm_name: str, *, standing: clearings.Standing) -> None:
    """One (block, arm) end to end (the module's docstring)."""
    b, name = propose(rows, bid, arm_name, standing=standing)
    scratch = rows / "scratch"
    rr._BLOCKS[bid] = b
    if rr.prefixes_one(rr.PrefixJob(rows, scratch, bid, name)) is not None:
        raise SystemExit(f"{bid} {name}: its prefixes failed")
    rr.cx_network(rows, scratch, _metric(), b, name, standing=standing)


def propose(rows: Path, bid: str, arm_name: str, *,
            standing: clearings.Standing) -> tuple[Block, str]:
    """`run`'s proposal half alone (the network, its steps and its proposal row; no scoring):
    the block and the network's name. A proposal `standing` answers is done (Keep: its row
    stands, and its network, written before the row) is not made again; the block and the name
    are returned for the scoring all the same."""
    rr._check_single_threaded()
    method = method_of(ARMS[arm_name])
    name = network_name(arm_name, method)
    scratch = rows / "scratch"
    row_path = rr.proposal_row(rows, bid, name, False)
    todo = standing.todo(row_path)
    [b] = common.build_blocks([bid])
    if not todo:
        print(f"{bid} {name}: its proposal row stands ({row_path})", flush=True)
        return b, name
    print(f"{bid} n={len(b.buildings)} {arm_name} as {name}: {method}", flush=True)
    sp = steps_path(rows, bid, name)
    sp.parent.mkdir(parents=True, exist_ok=True)
    # Every step to a name no pull matches (cluster pulls take *.parquet and never replace a local
    # file, so a log pulled mid-run would stay stale); the *.parquet once, when the run ends.
    partial = sp.with_name(sp.name + ".partial")

    def sink(df: pd.DataFrame) -> None:
        rr.write_parquet(df, partial)

    progress = scratch / "progress" / f"{bid}__{name}.json"
    with rr.Sampler(progress, every=0.5, write_every=30.0) as s, \
            arterial_steps.StepLog(sink) as log:
        prop = method.propose(b)
    rr.write_parquet(log.frame(), sp)
    partial.unlink(missing_ok=True)
    roads = prop.roads
    assert roads is not None
    net = rr.net_path(scratch, bid, name, False)
    net.parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(roads, net)
    row = dict(block=bid, n=len(b.buildings), area_km2=b.boundary.area / 1e6, method=name,
               arm=arm_name, status="ok", wall_s=s.wall, cpu_s=s.cpu, peak_rss_gb=s.peak,
               commits=prop.params["commits"], stop=prop.params["stop"],
               budget=prop.params["budget"], displaced=prop.params["displaced"],
               n_roads=len(roads),
               road_m=float(roads.geometry.length.sum()), steps=len(log.rows),
               host=socket.gethostname())
    standing.write(pd.DataFrame([row]), row_path)
    print(f"{bid} {name}: {row['commits']} commits ({row['stop']}), {len(roads)} roads, "
          f"{s.wall:.0f} s wall, peak RSS {s.peak:.2f} GB", flush=True)
    return b, name


# ------------------------------------------------------------------ report

class Sources(NamedTuple):
    """Where one network's rows live: its rows root and its name there."""
    rows: Path
    name: str


def _arm_sources(rows: Path) -> dict[str, Sources]:
    return {a: Sources(rows, network_name(a, method_of(arm))) for a, arm in ARMS.items()}


class Gap(NamedTuple):
    """One network's D by area under the three rules."""
    partial: float          # the overlap fraction (the public package's, the lineup's before)
    ramped: float           # common.MEASURE, the lineup's now
    touched: float          # a building the corridor touches at all counted whole


def _gap(b: Block, roads: gpd.GeoDataFrame) -> Gap:
    if not len(roads):
        return Gap(0.0, 0.0, 0.0)
    corridor = road_corridor(roads)
    c = b.buildings.displacement(corridor)
    area = np.asarray(b.buildings.outlines.area, dtype=np.float64)
    return Gap(partial=float((c * area).sum() / area.sum()),
               ramped=common.MEASURE.share(b.buildings, corridor, area),
               touched=float((area * (c > 0)).sum() / area.sum()))


def _run_row(b: Block, src: Sources, net: str) -> dict[str, object]:
    bid = b.block_id
    prop = pd.read_parquet(rr.proposal_row(src.rows, bid, src.name, False)).iloc[0]
    roads = gpd.read_parquet(rr.net_path(src.rows / "scratch", bid, src.name, False))
    pre = pd.read_parquet(rr.prefix_dir(src.rows) / f"{bid}__{src.name}.parquet")
    full = pre[pre.full & (pre["pop"] == "area")].iloc[0]
    gap = _gap(b, roads)
    r: dict[str, object] = dict(block=bid, area_km2=b.boundary.area / 1e6, net=net,
                                wall_s=float(prop.wall_s), n_roads=len(roads),
                                road_m=float(roads.geometry.length.sum()),
                                D_partial=gap.partial, D_ramped=gap.ramped,
                                D_touched=gap.touched,
                                D_count=float(full.D_count), D_area=float(full.D_area),
                                peak_rss_gb=float(prop.peak_rss_gb))
    if net == "baseline":
        r |= dict(commits=np.nan, stop="", host="")
    else:
        # The run's own D at its stop (the engine's incremental meter) beside the same network
        # scored afresh (the prefixes' whole-network D_area): the two must agree to FP, which is
        # the meter's check on real blocks.
        r |= dict(commits=int(prop.commits), stop=str(prop.stop), host=str(prop.host),
                  budget=str(prop.budget), displaced=float(prop.displaced),
                  chk_displaced=abs(float(prop.displaced) - float(full.D_area)))
        st = pd.read_parquet(steps_path(src.rows, bid, src.name))
        r |= dict(step_s_median=float(st.step_s.median()),
                  step_s_p90=float(st.step_s.quantile(0.9)), step_s_max=float(st.step_s.max()),
                  anchors_max=int(st.anchors.max()), candidates_max=int(st.candidates.max()))
    return r


def _lens_rows(src: Sources, bid: str, net: str, metric: gcl.ComplexMetric) -> pd.DataFrame:
    t = pd.read_parquet(rr.cx_dir(src.rows, metric) / f"{bid}__{src.name}.parquet")
    t = t.assign(net=net, budget=np.where(t.full, np.inf, t.budget))
    t["reach"] = t.full | (t.D_area >= t.budget - 1e-9)
    return pd.DataFrame(t[["block", "net", "budget", "full", "m", "road_m", "D_area", "reach",
                           *VERDICT_LENSES, "enclosed"]])


def _geoms(path: Path) -> NDArray[np.object_]:
    """A stored network's roads, in its order."""
    return np.asarray(gpd.read_parquet(path).geometry.to_numpy(), dtype=object)


def _same(a: NDArray[np.object_], b: NDArray[np.object_]) -> bool:
    """Road for road identical (geom_equals_exact at tolerance 0, as road_rescore's
    same_as_cached)."""
    return len(a) == len(b) and bool(shapely.equals_exact(a, b, tolerance=0.0).all())


def _fidelity(b: Block, rows: Path, name: str, base: Path, stop: str) -> dict[str, object]:
    """cur02 inside the baseline (length outside its buffered union), and equal to it where it
    stopped on no_gain."""
    bid = b.block_id
    cur = _geoms(rr.net_path(rows / "scratch", bid, name, False))
    ref = _geoms(rr.net_path(base / "scratch", bid, BASE_NET, False))
    outside = float(shapely.union_all(cur).difference(
        shapely.union_all(ref).buffer(FIDELITY_BUFFER)).length) if len(cur) else 0.0
    same = _same(cur, ref)
    return dict(fid_outside_m=outside, fid_inside=outside <= FIDELITY_TOL,
                fid_equal=same if stop == "no_gain" else np.nan)


def _prefix_rows(b: Block, rows: Path, name: str, base: Path) -> list[dict[str, object]]:
    bid = b.block_id
    out = []
    pre = {k: pd.read_parquet(rr.prefix_dir(r) / f"{bid}__{n}.parquet")
           for k, (r, n) in dict(cur=(rows, name), base=(base, BASE_NET)).items()}
    ordered = {k: _geoms(rr.ordered_path(r / "scratch", bid, n))
               for k, (r, n) in dict(cur=(rows, name), base=(base, BASE_NET)).items()}
    for budget in rr.BUDGETS:
        m = {k: int(p[(p["pop"] == "area") & np.isclose(p.budget, budget)].m.iloc[0])
             for k, p in pre.items()}
        u = {k: shapely.union_all(ordered[k][:m[k]]) for k in m}
        sd = float(u["cur"].symmetric_difference(u["base"]).length)
        same = _same(ordered["cur"][:m["cur"]], ordered["base"][:m["base"]])
        out.append(dict(block=bid, budget=budget, m_cur02=m["cur"], m_base=m["base"],
                        symdiff_m=sd, symdiff_share=sd / max(float(u["base"].length), 1e-9),
                        identical=same))
    return out


def _verdict(runs: pd.DataFrame, lenses: pd.DataFrame) -> list[str]:
    """Ruling 7 for SAMPLERS (the module's docstring): the lines to print."""
    a, b = SAMPLERS
    w = runs.pivot(index="block", columns="net", values="wall_s")
    if a not in w or b not in w:
        return [f"verdict: needs both {a} and {b}"]
    w = pd.DataFrame(w[[a, b]]).dropna()
    lines = [f"walls {b}/{a}: "
             + ", ".join(f"{k} {w.loc[k, b] / w.loc[k, a]:.2f}" for k in w.index)]
    med: dict[str, dict[tuple[str, float], float]] = {a: {}, b: {}}
    for budget in rr.REPORT_BUDGETS:
        at = lenses[np.isclose(lenses.budget, budget) & lenses.reach]
        for lens in VERDICT_LENSES:
            p = at.pivot(index="block", columns="net", values=lens)
            if a not in p or b not in p:
                continue
            d = (p[b] - p[a]).dropna()
            m = float(d.median()) if len(d) else np.nan
            med[b][lens, budget], med[a][lens, budget] = m, -m
            lines.append(f"  {lens} at D {budget:g}: {b} - {a} median {m:+.5f} over {len(d)} "
                         f"blocks; {b} ahead on {int((d > 0).sum())} "
                         f"({', '.join(d.index[d > 0])}), behind on {int((d < 0).sum())} "
                         f"({', '.join(d.index[d < 0])})")
    for x, y in ((b, a), (a, b)):
        faster = bool((w[x] < w[y]).all())
        ahead = (len(med[x]) == len(VERDICT_LENSES) * len(rr.REPORT_BUDGETS)
                 and all(v >= 0 for v in med[x].values()))
        lines.append(f"{x} faster on every block: {faster}; its six medians all >= 0: {ahead}"
                     f" -> {x} {'DOMINATES' if faster and ahead else 'does not dominate'} {y}")
    return lines


def report(rows: Path, base: Path, ids: list[str]) -> None:
    metric = _metric()
    srcs = {"baseline": Sources(base, BASE_NET), **_arm_sources(rows)}
    blocks = {b.block_id: b for b in common.build_blocks(ids)}
    run_rows, lens_frames, prefix_rows = [], [], []
    for bid in ids:
        b = blocks[bid]
        for net, src in srcs.items():
            if not rr.proposal_row(src.rows, bid, src.name, False).exists():
                continue
            r = _run_row(b, src, net)
            if net == "cur02":
                r |= _fidelity(b, rows, src.name, base, str(r["stop"]))
                prefix_rows += _prefix_rows(b, rows, src.name, base)
            run_rows.append(r)
            lens_frames.append(_lens_rows(src, bid, net, metric))
    runs = pd.DataFrame(run_rows)
    lenses = pd.concat(lens_frames, ignore_index=True)
    ref = lenses.loc[lenses.net == "baseline", ["block", "budget", *VERDICT_LENSES]]
    lenses = lenses.merge(ref, on=["block", "budget"], how="left", suffixes=("", "_base"),
                          validate="many_to_one")
    for lens in VERDICT_LENSES:
        lenses[f"d_{lens}"] = lenses[lens] - lenses.pop(f"{lens}_base")
    OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT / "summary_runs.csv", index=False)
    lenses.to_csv(OUT / "summary_lenses.csv", index=False)
    pd.DataFrame(prefix_rows).to_csv(OUT / "summary_prefixes.csv", index=False)
    pd.set_option("display.width", 250)
    print(runs.to_string(), flush=True)
    print(lenses.to_string(), flush=True)
    if prefix_rows:
        print(pd.DataFrame(prefix_rows).to_string(), flush=True)
    arms = runs[runs.net != "baseline"]
    print(f"meter against a fresh score: max |D| difference {arms.chk_displaced.max():.2e}",
          flush=True)
    for line in _verdict(runs, lenses):
        print(line, flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    a, standing = clearings.standing_of(argv)
    if len(a) == 4 and a[0] == "run":
        run(Path(a[1]), common.ids_of(a[2])[0], a[3], standing=standing)
    elif len(a) == 4 and a[0] == "propose":
        propose(Path(a[1]), common.ids_of(a[2])[0], a[3], standing=standing)
    elif len(argv) == 4 and argv[0] == "report":        # writes no row: takes no Standing's flag
        report(Path(argv[1]), Path(argv[2]), common.ids_of(argv[3]))
    else:
        raise SystemExit(__doc__)
