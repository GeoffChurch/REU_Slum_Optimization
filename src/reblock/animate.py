"""Prefix sweeps over a method's roads added in drainage order: render them into a per-method GIF
(`reblock_gif`) or read the region's max access-depth at each budget (`depth_sweep`).

Every prefix is an independent access peel, so the sweep runs across a fork `ProcessPoolExecutor`
(a 16-frame GIF collapses to ~one frame's wall-clock). The block is shared into workers by fork
inheritance (`_CTX` set before the pool), never pickled per prefix."""
from __future__ import annotations

import io
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from PIL import Image

from reblock.budget import _drainage_ordered
from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.emit import _displaced_points
from reblock.render import render_after

_T = TypeVar("_T")
_CTX: dict[str, Any] = {}   # fork-inherited per-sweep state (block is NOT pickled per prefix)


def _prefixes(block: Block, roads: GeoDataFrame, n: int,
              tol: float) -> tuple[GeoDataFrame, NDArray[np.float64], NDArray[np.float64]]:
    ordered = _drainage_ordered(block, roads, tol)
    cumlen = ordered.geometry.length.cumsum().to_numpy()
    cutoffs = np.linspace(0.0, float(cumlen[-1]), n)
    return ordered, cumlen, cutoffs


def _prefix_at(cutoff: float) -> tuple[int, GeoDataFrame]:
    k = int(np.searchsorted(_CTX["cumlen"], cutoff, side="right"))
    return k, cast(GeoDataFrame, _CTX["ordered"].iloc[:k])


def _run_parallel(worker: Callable[[tuple[int, float]], _T],
                  tasks: list[tuple[int, float]]) -> list[_T]:
    workers = min(len(tasks), max(1, (os.cpu_count() or 2) - 1))
    if workers > 1 and "fork" in multiprocessing.get_all_start_methods():
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("fork")) as ex:
            return list(ex.map(worker, tasks))
    return [worker(t) for t in tasks]


def _frame_png(task: tuple[int, float]) -> tuple[int, bytes]:
    """Render one GIF frame: the drainage-ordered prefix up to `cutoff` metres, parcels coloured by
    access depth under it. Returns (index, PNG bytes)."""
    import matplotlib.pyplot as plt
    idx, cutoff = task
    k, prefix = _prefix_at(cutoff)
    block: Block = _CTX["block"]
    proposal = Proposal(block_id=block.block_id, crs=block.crs, roads=prefix)
    layers = parcel_access_layers(block, prefix if k else None)
    fig = render_after(block, proposal, layers, vmax=_CTX["vmax"], frame=_CTX["frame"],
                       displaced_points=_displaced_points(block, proposal) if k else None)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_CTX["dpi"])   # fixed frame => fixed pixel size (GIF-safe)
    plt.close(fig)
    return idx, buf.getvalue()


def _max_depth(task: tuple[int, float]) -> tuple[int, float]:
    idx, cutoff = task
    k, prefix = _prefix_at(cutoff)
    layers = parcel_access_layers(_CTX["block"], prefix if k else None)
    return idx, (float(layers.max()) if len(layers) else 0.0)


def reblock_gif(block: Block, roads: GeoDataFrame, out_path: Path, *, vmax: int,
                frame: tuple[float, float, float, float], frames: int = 16,
                tol: float = STREET_TOL, dpi: int = 68, hold_last: int = 4) -> None:
    """Write a GIF of `roads` added to `block` in drainage order over `frames` cumulative-length
    budgets, on the shared access-depth scale `vmax` and fixed `frame` extent. No-op for empty
    roads. Frames render across a fork pool."""
    if roads is None or len(roads) == 0:
        return
    ordered, cumlen, cutoffs = _prefixes(block, roads, frames, tol)
    _CTX.update(block=block, ordered=ordered, cumlen=cumlen, vmax=vmax, frame=frame, dpi=dpi)
    rendered = _run_parallel(_frame_png, list(enumerate(cutoffs)))
    imgs = [Image.open(io.BytesIO(png)).convert("P", palette=Image.Palette.ADAPTIVE)
            for _, png in sorted(rendered)]
    durations = [220] * len(imgs)
    durations[-1] = 220 * hold_last                  # hold the finished network before looping
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=durations,
                 loop=0, optimize=True, disposal=2)


def depth_sweep(block: Block, roads: GeoDataFrame, *, n_points: int = 24,
                tol: float = STREET_TOL) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """`(cutoffs, max_depths)`: the region's MAX access-depth as `roads` is added in drainage order,
    at `n_points` cumulative-length budgets. Empty roads -> a single (0, max-depth) sample."""
    if roads is None or len(roads) == 0:
        base = parcel_access_layers(block, None)
        d = float(base.max()) if len(base) else 0.0
        return np.array([0.0]), np.array([d])
    ordered, cumlen, cutoffs = _prefixes(block, roads, n_points, tol)
    _CTX.update(block=block, ordered=ordered, cumlen=cumlen)
    res = _run_parallel(_max_depth, list(enumerate(cutoffs)))
    depths = np.array([d for _, d in sorted(res)], dtype=np.float64)
    return cutoffs, depths
