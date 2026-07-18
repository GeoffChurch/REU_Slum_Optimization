"""Per-method reblock animation: a GIF that adds a method's roads in drainage order and re-scores
parcel access-depth at each of `frames` cumulative-length budgets, rendering the region on the same
shared colour scale as the static after-images -- so the deep-red interior visibly "drains" to blue
as the network reaches in.

Each frame is an independent access peel + render, so they render across a fork pool (a 16-frame GIF
collapses to ~one frame's wall-clock on a multi-core box). The block is shared into the workers by
fork inheritance (`_CTX` set before the pool), never pickled per frame."""
from __future__ import annotations

import io
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
from geopandas import GeoDataFrame
from PIL import Image

from reblock.budget import _drainage_ordered
from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.emit import _displaced_points
from reblock.render import render_after

_CTX: dict[str, Any] = {}   # fork-inherited per-GIF state (block is NOT pickled per frame)


def _frame_png(task: tuple[int, float]) -> tuple[int, bytes]:
    """Render one frame: the drainage-ordered road prefix up to `cutoff` metres, the parcels
    coloured by their access depth under that prefix. Returns (index, PNG bytes)."""
    import matplotlib.pyplot as plt
    idx, cutoff = task
    block: Block = _CTX["block"]
    ordered: GeoDataFrame = _CTX["ordered"]
    cumlen: np.ndarray = _CTX["cumlen"]
    k = int(np.searchsorted(cumlen, cutoff, side="right"))
    prefix = cast(GeoDataFrame, ordered.iloc[:k])
    proposal = Proposal(block_id=block.block_id, crs=block.crs, roads=prefix)
    layers = parcel_access_layers(block, prefix if k else None)
    fig = render_after(block, proposal, layers, vmax=_CTX["vmax"], frame=_CTX["frame"],
                       displaced_points=_displaced_points(block, proposal) if k else None)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_CTX["dpi"])   # fixed frame => fixed pixel size (GIF-safe)
    plt.close(fig)
    return idx, buf.getvalue()


def reblock_gif(block: Block, roads: GeoDataFrame, out_path: Path, *, vmax: int,
                frame: tuple[float, float, float, float], frames: int = 16,
                tol: float = STREET_TOL, dpi: int = 68, hold_last: int = 4) -> None:
    """Write a GIF of `roads` added to `block` in drainage order over `frames` cumulative-length
    budgets, on the shared access-depth scale `vmax` and fixed `frame` extent. No-op for empty
    roads. Frames render across a fork pool."""
    if roads is None or len(roads) == 0:
        return
    ordered = _drainage_ordered(block, roads, tol)
    cumlen = ordered.geometry.length.cumsum().to_numpy()
    cutoffs = np.linspace(0.0, float(cumlen[-1]), frames)
    _CTX.update(block=block, ordered=ordered, cumlen=cumlen, vmax=vmax, frame=frame, dpi=dpi)
    tasks = list(enumerate(cutoffs))
    workers = min(frames, max(1, (os.cpu_count() or 2) - 1))
    if workers > 1 and "fork" in multiprocessing.get_all_start_methods():
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("fork")) as ex:
            rendered = list(ex.map(_frame_png, tasks))
    else:
        rendered = [_frame_png(t) for t in tasks]
    imgs = [Image.open(io.BytesIO(png)).convert("P", palette=Image.Palette.ADAPTIVE)
            for _, png in sorted(rendered)]
    durations = [220] * len(imgs)
    durations[-1] = 220 * hold_last                  # hold the finished network before looping
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=durations,
                 loop=0, optimize=True, disposal=2)
