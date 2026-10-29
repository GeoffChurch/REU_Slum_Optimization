"""How much faster our fork of the original topology algorithm is than the original (owner
2026-10-29: does "topology" on the site mean the original's time or our port's?).

The fork is ext/topology as pinned (ff74af2: cached node and edge hashes, an inverted-index weak
dual in place of the all-pairs face comparison, a Dijkstra-based find_short_paths_all_parcels and
its order-preserving prune, all in 25a69d3 and ff74af2). The original is ORIGINAL (5972772), the
last commit before any of them: the Python 3 port (5de88c9) with only correctness fixes on top
(6beb54a's package layout, 584669c's k-complexity, 5972772's outer-face selection, which can change
a network: comparing against 6beb54a would mix a correctness change into the speed-up). It is
checked out in its own worktree of the submodule (the pinned checkout untouched) and put first on
each run's PYTHONPATH (the venv's editable install comes after it; each run records the module's
file).

Each (block, version) is one process: TopologyMethod(alpha 10, seed 0, 7 m roads), one thread
(NUMBA_NUM_THREADS=1 and the BLAS threads 1), its wall time, under a hard timeout (TIMEOUT_S: a run
that hits it is censored), WORKERS at a time. Blocks: Cape Town blocks of 20 -- 150 buildings, the
three of build.py's timed six in that range (ZAF.9.3.1_1_38136, _21016, _20254: 51, 93, 122) and
ten more, the first block (by id) nearest each of TARGETS buildings (block_sizes.parquet; the
135- and the 140-building ones raise in both versions, "parcel ... is on a road": kept as rows).
Then per block: whether the two networks are the same segment set (topology_order.same_network)
and, where they differ, each one's D and L_P (road_simp.scored, RUNTIME).

    S=research/roadless/atlas/topology_speed.py        # from the repo root, PYTHONPATH=.
    uv run python $S run <original worktree>           # every block, both versions
    uv run python $S table                             # compare, fit, write OUT/table.csv
    uv run python $S one <block> <version> <out>       # one run (what `run` starts)
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROADLESS = HERE.parent
ROOT = HERE.parents[2]
OUT = HERE / "topology_speed"            # committed: per run a json and its roads
FORK, ORIGINAL = "ff74af2", "5972772"
TIMEOUT_S = 2400                         # a run past 40 min is censored
WORKERS = 8
FIT_BLOCKS = ("ZAF.9.3.1_1_38136", "ZAF.9.3.1_1_21016", "ZAF.9.3.1_1_20254")
TARGETS = (20, 30, 40, 60, 75, 85, 105, 135, 140, 150)


def blocks() -> list[tuple[str, int]]:
    d = pd.read_parquet(ROADLESS / "block_sizes.parquet")
    d = d[d.block.str.startswith("ZAF.9.3.1_1_")].sort_values("block")
    out = {b: int(d.set_index("block").n[b]) for b in FIT_BLOCKS}
    for t in TARGETS:
        rest = d[~d.block.isin(list(out))]
        i = (rest.n - t).abs().idxmin()
        out[str(rest.block[i])] = int(rest.n[i])
    return sorted(out.items(), key=lambda kv: kv[1])


def one(bid: str, version: str, out: Path) -> None:
    """One timed run in this process (`run` starts it with the version's PYTHONPATH)."""
    sys.path.insert(0, str(ROADLESS))
    import common  # noqa: PLC0415
    import topology  # noqa: PLC0415

    from reblock.methods.topology import TopologyMethod  # noqa: PLC0415
    [b] = common.build_blocks([bid])
    t0, c0 = time.time(), time.process_time()
    p = TopologyMethod(alpha=10.0, seed=0, road_width_m=7.0).propose(b)
    wall, cpu = time.time() - t0, time.process_time() - c0
    assert p.roads is not None
    p.roads.to_parquet(out.with_suffix(".parquet"))
    out.write_text(json.dumps(dict(block=bid, version=version, n_buildings=len(b.buildings),
                                   wall_s=wall, cpu_s=cpu, n_roads=len(p.roads),
                                   road_m=float(p.roads.geometry.length.sum()),
                                   module=topology.__file__, host=socket.gethostname()),
                              indent=1) + "\n")


def launch(bid: str, version: str, path: str | None) -> None:
    out = OUT / f"{bid}_{version}.json"
    if out.exists():
        return
    env = os.environ | dict(NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                            MKL_NUM_THREADS="1",
                            PYTHONPATH=":".join(([path] if path else []) + [str(ROOT)]))
    cmd = ["uv", "run", "python", str(Path(__file__)), "one", bid, version, str(out)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, cwd=ROOT, timeout=TIMEOUT_S, capture_output=True,
                           text=True)
    except subprocess.TimeoutExpired:
        out.write_text(json.dumps(dict(block=bid, version=version, censored=True,
                                       wall_s=time.time() - t0)) + "\n")
        print(f"{bid} {version}: censored past {TIMEOUT_S} s", flush=True)
        return
    if r.returncode != 0:          # the run raised: recorded, its last line the reason
        why = [x for x in r.stderr.splitlines() if x.strip()][-1]
        out.write_text(json.dumps(dict(block=bid, version=version, failed=why,
                                       wall_s=time.time() - t0)) + "\n")
        print(f"{bid} {version}: failed after {time.time() - t0:.1f} s: {why}", flush=True)
        return
    print(f"{bid} {version}: {json.loads(out.read_text())['wall_s']:.1f} s", flush=True)


def run(original: Path) -> None:
    OUT.mkdir(exist_ok=True)
    bs = blocks()
    print(f"{len(bs)} blocks: {bs}", flush=True)
    jobs = [(b, v, p) for b, _ in bs for v, p in ((ORIGINAL, str(original)), (FORK, None))]
    jobs.sort(key=lambda j: -dict(bs)[j[0]])          # the biggest first: the longest
    with ThreadPoolExecutor(WORKERS) as ex:
        for f in [ex.submit(launch, *j) for j in jobs]:
            f.result()


def table() -> None:
    sys.path.insert(0, str(ROADLESS))
    sys.path.insert(0, str(HERE))
    import geopandas as gpd  # noqa: PLC0415
    import topology_order as to  # noqa: PLC0415
    rows = []
    for b, n in blocks():
        o = json.loads((OUT / f"{b}_{ORIGINAL}.json").read_text())
        f = json.loads((OUT / f"{b}_{FORK}.json").read_text())
        if "ext/topology/" not in f.get("module", "ext/topology/") or (
                "module" in o and "ext/topology/" in o["module"]):
            raise AssertionError(f"{b}: a run imported the wrong topology ({o}, {f})")
        row = dict(block=b, n=n, original_s=o["wall_s"], fork_s=f["wall_s"],
                   censored=bool(o.get("censored", False)), failed=o.get("failed", ""),
                   speedup=o["wall_s"] / f["wall_s"])
        if "failed" in f:              # both versions raise on it (the vendored algorithm's)
            rows.append(dict(block=b, n=n, failed=f"both: {f['failed']}"))
            continue
        if not o.get("censored") and "failed" not in o:
            ro = gpd.read_parquet(OUT / f"{b}_{ORIGINAL}.parquet")
            rf = gpd.read_parquet(OUT / f"{b}_{FORK}.parquet")
            row |= dict(same=to.same_network(ro, rf), roads_original=len(ro), roads_fork=len(rf))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table.csv", index=False)
    ok = ~df.censored.fillna(True).astype(bool) & (df.failed.fillna("") == "")
    x = np.log(df.n[ok].to_numpy(float))
    for col in ("original_s", "fork_s"):
        b_, a_ = np.polyfit(x, np.log(df[col][ok].to_numpy(float)), 1)
        print(f"{col}: seconds ~ n^{b_:.2f}")
    b_, _ = np.polyfit(x, np.log(df.speedup[ok].to_numpy(float)), 1)
    print(f"speed-up ~ n^{b_:.2f}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["run", original]:
            run(Path(original))
        case ["table"]:
            table()
        case ["one", bid, version, out]:
            one(bid, version, Path(out))
        case _:
            raise SystemExit(__doc__)
