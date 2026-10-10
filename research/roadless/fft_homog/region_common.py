"""Shared by the region_*.py scripts: the deciding experiment for FFT homogenization on
ZAF.9.3.1_1_5810@major (report.md, sections 3.6 and 6).

Paths, the thread settings every process must run with, a cost monitor (wall, CPU, peak
proportional set size of the process tree), and the region's fabric helpers. Large outputs go
to OUT (the session's scratch), never into the repo.
"""
from __future__ import annotations

import csv
import os
import resource
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                      # research/roadless/fft_homog
ROADLESS = HERE.parent
REPO = ROADLESS.parent.parent
OUT = Path(os.environ.get(
    "FFT_REGION_OUT",
    "/tmp/claude-1641171234/-home-gchurchill-src-reblock/fe8870c0-fbd9-4712-ac98-aebcb951b199/"
    "scratchpad/fft_homog_region"))
BID = "ZAF.9.3.1_1_5810@major"
H = 0.5                    # the metric's spacing (m per px)
CHEAP = (ROADLESS / "polish_rows/uni/S0.01cat.P64w8.a5x8/D0.05"
         / "ZAF.9.3.1_1_5810@major_p2.parquet")
SIMP = (ROADLESS / "relax_rows/uni/fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2.a5x8/D0.05"
        / "ZAF.9.3.1_1_5810@major_p2.parquet")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")

for _v in THREAD_VARS:
    if os.environ.get(_v) != "1":
        raise SystemExit(f"{_v} must be 1 (the machine is shared)")
for _p in (str(HERE), str(ROADLESS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ------------------------------------------------------------------------------ costs
def _children(pid: int, table: dict[int, list[int]]) -> list[int]:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += table.get(p, [])
    return out


def tree_pss_kb(root: int) -> int:
    """Proportional set size summed over `root` and its live descendants (shared copy-on-write
    pages counted once in total)."""
    table: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        table.setdefault(ppid, []).append(int(d))
    tot = 0
    for p in _children(root, table):
        try:
            with open(f"/proc/{p}/smaps_rollup") as f:
                for line in f:
                    if line.startswith("Pss:"):
                        tot += int(line.split()[1])
                        break
        except OSError:
            continue
    return tot


def _cpu_s() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


@dataclass
class Cost:
    piece: str
    wall_s: float
    cpu_s: float
    peak_gb: float
    note: str


class Monitor:
    """`with Monitor("piece"):` -- wall time, CPU time of this process plus its reaped children
    (a pool must be shut down inside the block to be counted), and the peak PSS of the process
    tree sampled every `period` s. Appends a row to OUT/costs.csv."""

    def __init__(self, piece: str, note: str = "", period: float = 1.0):
        self.piece, self.note, self.period = piece, note, period
        self.peak = 0
        self._stop = threading.Event()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, tree_pss_kb(os.getpid()))
            self._stop.wait(self.period)

    def __enter__(self) -> Monitor:
        self.t0, self.c0 = time.perf_counter(), _cpu_s()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._th.join()
        self.peak = max(self.peak, tree_pss_kb(os.getpid()))
        self.cost = Cost(self.piece, time.perf_counter() - self.t0, _cpu_s() - self.c0,
                         self.peak / 2 ** 20, self.note)
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "costs.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["piece", "wall_s", "cpu_s", "peak_gb", "note"])
            w.writerow([self.cost.piece, f"{self.cost.wall_s:.1f}", f"{self.cost.cpu_s:.1f}",
                        f"{self.cost.peak_gb:.2f}", self.cost.note])
        print(f"[cost] {self.cost.piece}: wall {self.cost.wall_s:.1f} s, cpu {self.cost.cpu_s:.1f}"
              f" s, peak {self.cost.peak_gb:.2f} GB {self.note}", flush=True)


# ------------------------------------------------------------------------------ fabric
@dataclass
class Fabric:
    """The region on the uniform h 0.5 lattice (lifted.UniformMesh(0.5, offset=lifted.OFFSET)),
    as region_extract.py wrote it."""
    o: np.ndarray            # (ny, nx) open fraction ff0
    inside: np.ndarray
    ground: np.ndarray
    f: np.ndarray            # injection (lifted.demand, area population)
    owner: np.ndarray        # building per ring cell, -1 none
    w: np.ndarray            # per building: population weight
    live: np.ndarray         # per building: not stranded
    area: np.ndarray         # per building: footprint m^2
    cy: np.ndarray           # per building: mean ring-cell row / column (pick.py's), cnt ring
    cx: np.ndarray           #   cells (0: no ring cell)
    cnt: np.ndarray
    py: np.ndarray           # per building: polygon centroid row / column (px)
    px: np.ndarray
    sc: np.ndarray           # (cell, building, count) of footprint sub-samples, by building
    sb: np.ndarray
    sn: np.ndarray
    starts: np.ndarray       # sb's first index per building (len n + 1)


def load_fabric() -> Fabric:
    d = np.load(OUT / "fabric_region.npz")
    order = np.argsort(d["sub_bldg"], kind="stable")
    sb = d["sub_bldg"][order]
    n = len(d["w"])
    return Fabric(o=d["ff0"].astype(np.float64), inside=d["inside"], ground=d["ground"],
                  f=d["f"], owner=d["owner"], w=d["w"], live=~d["stranded"], area=d["area"],
                  cy=d["cy"], cx=d["cx"], cnt=d["cnt"], py=d["py"], px=d["px"],
                  sc=d["sub_cell"][order], sb=sb, sn=d["sub_count"][order],
                  starts=np.searchsorted(sb, np.arange(n + 1)))


def open_buildings(fab: Fabric, ids) -> np.ndarray:
    """The open fraction with buildings `ids` cleared, exactly (their footprint sub-samples
    opened), as extract2.py's (isub & (~bsub | cleared)).mean()."""
    o = fab.o.copy().ravel()
    for b in np.asarray(ids, dtype=np.int64):
        a, z = fab.starts[b], fab.starts[b + 1]
        np.add.at(o, fab.sc[a:z], fab.sn[a:z] / 16.0)
    return np.minimum(o, 1.0).reshape(fab.o.shape)


def stored_clearings() -> dict[str, np.ndarray]:
    import pandas as pd
    cheap = pd.read_parquet(CHEAP)
    simp = pd.read_parquet(SIMP)
    return {"cheap": np.asarray(cheap.cleared.iloc[0], dtype=np.int64),
            "simp": np.asarray(simp.simp_cleared.iloc[0], dtype=np.int64)}


STORED_LENS_A = {"cheap": 0.271361, "simp": 0.301406}   # the parquet rows' perm / simp_perm


def full_tiles(inside: np.ndarray, n: int) -> list[tuple[int, int]]:
    """Tiles of n x n px aligned to the raster origin with every cell inside (tiles.py's)."""
    integ = np.pad(inside.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    out = []
    for r0 in range(0, inside.shape[0] - n + 1, n):
        for c0 in range(0, inside.shape[1] - n + 1, n):
            if (integ[r0 + n, c0 + n] - integ[r0, c0 + n] - integ[r0 + n, c0]
                    + integ[r0, c0]) == n * n:
                out.append((r0, c0))
    return out
