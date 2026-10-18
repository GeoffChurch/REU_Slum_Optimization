"""Per-step timing of the lineup's arterial (road_rescore.lineup_arms()[ARTERIAL]) on one block,
through the real engine loop (`engines._greedy_shortlist`), with timing wrappers patched into the
module namespaces the loop looks names up in. Nothing in src/ is edited.

usage: PYTHONPATH=. python steps.py <block_id> <max_roads or 0 for the lineup's> <out_dir>
Writes <out_dir>/steps_<bid>.csv (one row per greedy step, flushed as it goes), then the commit
order's displacement per step (commit_D_<bid>.csv), the fidelity check against the stored network,
and the truncation comparison (truncate_<bid>.csv).
"""
import csv
import dataclasses
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "research/roadless")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
import road_rescore  # noqa: E402
from reblock.derive import access  # noqa: E402
from reblock.methods.arterial import costs, engines, objectives, realize, shortlist  # noqa: E402
from reblock.methods.arterial.primitives import _planarize  # noqa: E402

bid, cap, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])   # name -> [inclusive s, calls]
info: dict[str, int] = {}
commits = []


def timed(name, fn, note=None):
    def w(*a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        e = acc[name]
        e[0] += time.perf_counter() - t
        e[1] += 1
        if note is not None:
            info[note] = note_of(note, r)
        return r
    return w


def note_of(note, r):
    if note == "segments":
        return len(r.roads)
    return len(r)


# the loop's own names (engines namespace)
engines._anchor_points = timed("anchors", engines._anchor_points, "anchors")
engines._step_network = timed("step_network", engines._step_network, "segments")
engines._deep_targets = timed("deep_targets", engines._deep_targets)
engines._candidate_chords = timed("candidate_chords", engines._candidate_chords, "candidates")
engines.parcel_access_layers = timed("rank_depths_peel", engines.parcel_access_layers)
engines.step_state = timed("step_state", engines.step_state)
engines._planarize = timed("final_planarize", engines._planarize)
engines.road_drainage = timed("final_drainage", engines.road_drainage)
engines.eval_candidate = timed("eval_candidate", engines.eval_candidate)
shortlist.FirstOrder.select = timed("shortlist_select", shortlist.FirstOrder.select, "shortlisted")
# inside eval_candidate
realize.SnapToBoundary.realize = timed("e_snap", realize.SnapToBoundary.realize)
objectives._Unioned.of = timed("e_gain", objectives._Unioned.of)
objectives._union_with = timed("e_gain_union", objectives._union_with)
objectives._explode = timed("e_gain_explode", objectives._explode)
objectives.parcel_access_layers = timed("e_gain_peel", objectives.parcel_access_layers)
access.street_connectivity = timed("street_connectivity(any)", access.street_connectivity)
costs._NewlyDisplaced.of = timed("e_cost", costs._NewlyDisplaced.of)

fields = ["step", "t_end", "step_s", "segments", "anchors", "candidates", "shortlisted",
          "committed"]
names = ["anchors", "step_network", "deep_targets", "candidate_chords", "rank_depths_peel",
         "shortlist_select", "step_state", "eval_candidate", "e_snap", "e_gain", "e_gain_union",
         "e_gain_explode", "e_gain_peel", "street_connectivity(any)", "e_cost"]
fh = open(out / f"steps_{bid}.csv", "w", newline="")
wr = csv.DictWriter(fh, fieldnames=fields + [f"{n}_s" for n in names] + [f"{n}_n" for n in names])
wr.writeheader()
T0 = [time.perf_counter()]
last = [T0[0]]
real_best = engines._best_candidate


def best(results):
    results = list(results)
    g, r = real_best(results)
    now = time.perf_counter()
    if r is not None:
        commits.append(r)
    row = dict(step=len(commits) - (r is not None), t_end=now - T0[0], step_s=now - last[0],
               committed=r is not None, segments=info.get("segments"),
               anchors=info.get("anchors"), candidates=info.get("candidates"),
               shortlisted=info.get("shortlisted"))
    for n in names:
        row[f"{n}_s"], row[f"{n}_n"] = acc[n][0], acc[n][1]
    wr.writerow(row)
    fh.flush()
    acc.clear()
    last[0] = now
    return g, r


engines._best_candidate = best

[b] = common.build_blocks([bid])
m = road_rescore.lineup_arms()[road_rescore.ARTERIAL]
if cap:
    m = dataclasses.replace(m, max_roads=cap)
print(m, flush=True)
t = time.perf_counter()
T0[0] = last[0] = t
prop = m.propose(b)
wall = time.perf_counter() - t
roads = prop.roads
print(f"{bid}: {len(commits)} commits, {len(roads)} segments, {wall:.0f}s; tail (final planarize "
      f"+ drainage) {acc['final_planarize'][0]:.1f}+{acc['final_drainage'][0]:.1f}s", flush=True)
fh.close()

# ---------------------------------------------------------------- after the timed run
stored = Path(f"research/roadless/road_rows/lineup-caf14fa2/scratch/networks/{bid}/"
              f"{road_rescore.ARTERIAL}.parquet")
if not cap and stored.exists():
    s = gpd.read_parquet(stored)
    same = len(s) == len(roads) and bool(s.geometry.reset_index(drop=True).geom_equals_exact(
        roads.geometry.reset_index(drop=True), tolerance=0.0).all())
    print(f"fidelity: identical to the lineup's stored network: {same}", flush=True)

polys = np.asarray(b.buildings.outlines)
W = {pop: common.POPULATIONS[pop].weights(polys) for pop in ("count", "area")}
crs, width = b.crs, 7.0
rows = []
for k in range(1, len(commits) + 1):
    pre = _planarize(commits[:k], crs, width)
    rows.append(dict(commits=k, segments=len(pre), road_m=float(pre.length.sum()),
                     D_count=common.displacement(b, pre, W["count"]),
                     D_area=common.displacement(b, pre, W["area"])))
cd = pd.DataFrame(rows)
cd.to_csv(out / f"commit_D_{bid}.csv", index=False)
print("commit-order D written", flush=True)

# truncation: stop the greedy once its commit-order area D reaches tau; are the scored canonical
# prefixes at budgets <= tau the same as the whole network's?
from reblock.budget import road_drainage  # noqa: E402

full = gpd.read_parquet(stored) if stored.exists() else roads
fcache = {}


def full_prefix(bud, pop):
    if (bud, pop) not in fcache:
        fcache[bud, pop] = common.prefix_to(b, full, bud, W[pop])
    return fcache[bud, pop]


trunc = []
for tau in (0.05, 0.10, 0.15, 0.20):
    hit = cd.index[cd.D_area >= tau]
    if not len(hit):
        continue
    k = int(cd.commits[hit[0]])
    for extra in (0, 10, 25):
        kk = min(len(commits), k + extra)
        net = _planarize(commits[:kk], crs, width)
        net["drain"] = road_drainage(b, net)
        for bud in (0.02, 0.05, 0.10, 0.15, 0.20):
            if bud > tau:
                continue
            for pop in ("count", "area"):
                a = common.prefix_to(b, net, bud, W[pop])
                f = full_prefix(bud, pop)
                ua, uf = a.geometry.union_all(), f.geometry.union_all()
                sd = ua.symmetric_difference(uf).length
                trunc.append(dict(tau=tau, commits_at_tau=k, extra=extra, commits=kk,
                                  budget=bud, pop=pop, len_trunc=ua.length, len_full=uf.length,
                                  symdiff_m=sd, symdiff_frac=sd / max(uf.length, 1e-9),
                                  D_trunc=common.displacement(b, a, W[pop]),
                                  D_full=common.displacement(b, f, W[pop])))
tr = pd.DataFrame(trunc)
tr.to_csv(out / f"truncate_{bid}.csv", index=False)
pd.set_option("display.width", 250)
print(tr.to_string(), flush=True)
